# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

from argparse import Namespace
from distutils.util import strtobool
import logging
import math

import numpy
import torch

from espnet.nets.pytorch_backend.e2e_asr_transformer import *

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding

from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.decoder_layer import DecoderLayer

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E as E2ETransformer
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.decoder import Decoder

from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from data_load import low_resource_languages
low_resource_adapter_dim = 64
high_resource_adapter_dim = 128

class AdapterFusion(MultiHeadedAttention):
    def __init__(self, n_feat, dropout_rate, fusion_languages=None, num_shared_layers=-1):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        self.linear_q = torch.nn.Linear(n_feat, n_feat)
        self.linear_k = torch.nn.Linear(n_feat, n_feat)
        self.linear_v = torch.nn.Linear(n_feat, n_feat, bias=False)
        self.linear_v.weight.data = (
            torch.zeros(n_feat, n_feat) + 0.000001
        ).fill_diagonal_(1.0)
        self.attn = None
        self.num_shared_layers = num_shared_layers
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.temperature = 1.0
        self.fusion_languages = fusion_languages

    def forward_qkv(self, query, key, value):
        q = self.linear_q(query) # (batch, time, d_k)
        k = self.linear_k(key) # (batch, time, n_adapters, d_k)
        v = self.linear_v(value) # (batch, time, n_adapters, d_k)
        return q, k, v

    def forward_attention(self, value, scores):
        self.attn = torch.softmax(scores, dim=-1)  # (batch, time, n_adapters)
        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn.unsqueeze(2), value)
        # (batch, time, 1, n_adapters), (batch, time, n_adapters, d_k)
        x = torch.squeeze(x, dim=2)
        return x  # (batch, time, d_k)

    def forward(self, query, key, value, residual=None):
        q, k, v = self.forward_qkv(query, key, value)
        # q: (batch, time, 1, d_k); k, v: (batch, time, n_adapters, d_k)
        scores = torch.matmul(q.unsqueeze(2), k.transpose(-2, -1)) / math.sqrt(self.temperature)
        scores = torch.squeeze(scores, dim=2)
        # scores: (batch, time, n_adapters)
        out = self.forward_attention(v, scores)
        if residual is not None:
            out = out + residual
        # self.temperature = max(self.temperature - self.temperature_reduction_steps, 1.0)
        return out

class Adapter(torch.nn.Module):
    def __init__(self, adapter_dim, embed_dim):
        super().__init__()
        self.layer_norm = LayerNorm(embed_dim)
        self.down_project = torch.nn.Linear(embed_dim, adapter_dim, bias=False)
        self.up_project = torch.nn.Linear(adapter_dim, embed_dim, bias=False)

    def forward(self, z):
        normalized_z = self.layer_norm(z)
        h = torch.nn.functional.relu(self.down_project(normalized_z))
        return self.up_project(h) + z
class CustomEncoderLayer(EncoderLayer):
    def forward(self, x, mask, cache=None):
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]
        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.self_attn(x_q, x, x, mask))
        if not self.normalize_before:
            x = self.norm1(x)
        residual = x
        if self.normalize_before:
            x = self.norm2(x)
            x_norm = x
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)
        if cache is not None:
            x = torch.cat([cache, x], dim=1)
        return x, x_norm, mask
class AdaptiveEncoderLayer(CustomEncoderLayer):
    def __init__(
        self,
        languages,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        adapter_fusion_layer=None,
        shared_adapter=None,
        use_adapters=True,
    ):
        super().__init__(size,
            self_attn,
            feed_forward,
            dropout_rate,
            normalize_before,
            concat_after,
        )
        self.use_adapters = use_adapters
        if use_adapters:
            self.adapters = torch.nn.ModuleDict()
            self.shared_adapter = shared_adapter
            if shared_adapter:
                languages = [shared_adapter]
            for lang in languages:
                if lang in low_resource_languages or self.shared_adapter:
                    adapter_dim = low_resource_adapter_dim
                else:
                    adapter_dim = high_resource_adapter_dim
                self.adapters[lang] = Adapter(adapter_dim, size)
            self.adapter_fusion = adapter_fusion_layer

    def forward(self, x, mask, language, cache=None, use_adapter_fusion=True):
        x, x_norm, mask = super().forward(x, mask, cache=cache)

        if not self.use_adapters:
            return x, mask, language, cache, use_adapter_fusion
        if self.shared_adapter:
            assert len(self.adapters.keys()) == 1
            language = list(self.adapters.keys())[0]
        if (not use_adapter_fusion) or not (self.adapter_fusion):
            out = self.adapters[language](x)
        else:
            out = []
            fusion_languages = list(self.adapter_fusion.keys())[0]
            for lang in fusion_languages.split("_"):
                if lang != "self":
                    out.append(self.adapters[lang](x))
                else:
                    out.append(x)
            out = torch.stack(out).permute(1, 2, 0, 3) # B, T, n_adapters, F
            out = self.adapter_fusion[fusion_languages](x, out, out, residual=x_norm)
            #out = self.adapters['cs'](x)
        return out, mask, language, cache, use_adapter_fusion

class AdaptiveEncoder(Encoder):
    def __init__(
        self,
        languages,
        idim,
        selfattention_layer_type="selfattn",
        attention_dim=256,
        attention_heads=4,
        conv_wshare=4,
        conv_kernel_length=11,
        conv_usebias=False,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer="conv2d",
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        padding_idx=-1,
        adapter_fusion=False,
        shared_adapter=None,
        use_adapters=True,
        fusion_languages=None,
    ):
        super().__init__(idim,
            selfattention_layer_type,
            attention_dim,
            attention_heads,
            conv_wshare,
            conv_kernel_length,
            conv_usebias,
            linear_units,
            num_blocks,
            dropout_rate,
            positional_dropout_rate,
            attention_dropout_rate,
            input_layer,
            pos_enc_class,
            normalize_before,
            concat_after,
            positionwise_layer_type,
            positionwise_conv_kernel_size,
            padding_idx)
        positionwise_layer, positionwise_layer_args = self.get_positionwise_layer(
            positionwise_layer_type,
            attention_dim,
            linear_units,
            dropout_rate,
            positionwise_conv_kernel_size,
        )
        if selfattention_layer_type == "selfattn":
            logging.info("encoder self-attention layer type = self-attention")
            self.encoders = repeat(
                num_blocks,
                lambda lnum: AdaptiveEncoderLayer(
                    languages,
                    attention_dim,
                    MultiHeadedAttention(
                        attention_heads, attention_dim, attention_dropout_rate
                    ),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                    torch.nn.ModuleDict({"_".join(sorted(fusion_languages)): AdapterFusion(attention_dim, attention_dropout_rate, fusion_languages)}) if adapter_fusion else None,
                    shared_adapter,
                    use_adapters,
                ),
            )
        else:
            raise NotImplementedError("Only support self-attention encoder layer")
    def forward(self, xs, masks, language, use_adapter_fusion=True):
        xs, masks = self.embed(xs, masks)
        xs, masks, _, _, _ = self.encoders(xs, masks, language, None, use_adapter_fusion)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks
class CustomDecoderLayer(DecoderLayer):
    def forward(self, tgt, tgt_mask, memory, memory_mask, cache=None):
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = None
            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[:, -1:, :]
        if self.concat_after:
            tgt_concat = torch.cat(
                (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1
            )
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        if not self.normalize_before:
            x = self.norm1(x)
        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        if self.concat_after:
            x_concat = torch.cat(
                (x, self.src_attn(x, memory, memory, memory_mask)), dim=-1
            )
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask))
        if not self.normalize_before:
            x = self.norm2(x)
        residual = x
        if self.normalize_before:
            x = self.norm3(x)
            x_norm = x
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)
        if cache is not None:
            x = torch.cat([cache, x], dim=1)
        return x, x_norm, tgt_mask, memory, memory_mask
class AdaptiveDecoderLayer(CustomDecoderLayer):
    def __init__(
        self,
        languages,
        size,
        self_attn,
        src_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        adapter_fusion_layer=None,
        shared_adapter=None,
        use_adapters=True,
    ):
        super().__init__(size,
        self_attn,
        src_attn,
        feed_forward,
        dropout_rate,
        normalize_before,
        concat_after
        )
        self.use_adapters = use_adapters
        if use_adapters:
            self.adapters = torch.nn.ModuleDict()
            self.shared_adapter = shared_adapter
            if shared_adapter:
                languages = [shared_adapter]
            for lang in languages:
                if lang in low_resource_languages or self.shared_adapter:
                    adapter_dim = low_resource_adapter_dim
                else:
                    adapter_dim = high_resource_adapter_dim
                self.adapters[lang] = Adapter(adapter_dim, size)
            self.adapter_fusion = adapter_fusion_layer
            
    def forward(self, tgt, tgt_mask, memory, memory_mask, language, cache=None, use_adapter_fusion=True):
        x, x_norm, tgt_mask, memory, memory_mask = super().forward(tgt, tgt_mask, memory, memory_mask, cache=cache)
        if not self.use_adapters:
            return x, tgt_mask, memory, memory_mask, language, cache, use_adapter_fusion
        if self.shared_adapter:
            assert len(self.adapters.keys()) == 1
            language = list(self.adapters.keys())[0]
        if (not use_adapter_fusion) or (not self.adapter_fusion):
            out = self.adapters[language](x)
        else:
            out = []
            fusion_languages = list(self.adapter_fusion.keys())[0]
            for lang in fusion_languages.split("_"):
                if lang != "self":
                    out.append(self.adapters[lang](x))
                else:
                    out.append(x)
            out = torch.stack(out).permute(1, 2, 0, 3) # B, T, n_adapters, F
            out = self.adapter_fusion[fusion_languages](x, out, out, residual=x_norm)
            
        return out, tgt_mask, memory, memory_mask, language, cache, use_adapter_fusion


class AdaptiveDecoder(Decoder):
    def __init__(
        self,
        languages,
        odim_dict,
        selfattention_layer_type="selfattn",
        attention_dim=256,
        attention_heads=4,
        conv_wshare=4,
        conv_kernel_length=11,
        conv_usebias=False,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        self_attention_dropout_rate=0.0,
        src_attention_dropout_rate=0.0,
        input_layer="embed",
        use_output_layer=True,
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
        adapter_fusion=False,
        shared_adapter=False,
        use_adapters=True,
        fusion_languages=None,
    ):
        super().__init__(1,
        selfattention_layer_type,
        attention_dim,
        attention_heads,
        conv_wshare,
        conv_kernel_length,
        conv_usebias,
        linear_units,
        num_blocks,
        dropout_rate,
        positional_dropout_rate,
        self_attention_dropout_rate,
        src_attention_dropout_rate,
        input_layer,
        use_output_layer,
        pos_enc_class,
        normalize_before,
        concat_after)
        if input_layer == "embed":
            self.embed = torch.nn.ModuleDict()
            for lang in odim_dict.keys():
                self.embed[lang] = torch.nn.Sequential(
                    torch.nn.Embedding(odim_dict[lang], attention_dim),
                    pos_enc_class(attention_dim, positional_dropout_rate),
                )
        else:
            raise NotImplementedError("only support embed embedding layer")
        assert self_attention_dropout_rate == src_attention_dropout_rate
        if selfattention_layer_type == "selfattn":
            logging.info("decoder self-attention layer type = self-attention")
            self.decoders = repeat(
                num_blocks,
                lambda lnum: AdaptiveDecoderLayer(
                    languages,
                    attention_dim,
                    MultiHeadedAttention(
                        attention_heads, attention_dim, self_attention_dropout_rate
                    ),
                    MultiHeadedAttention(
                        attention_heads, attention_dim, src_attention_dropout_rate
                    ),
                    PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                    torch.nn.ModuleDict({"_".join(sorted(fusion_languages)): AdapterFusion(attention_dim, self_attention_dropout_rate, fusion_languages)}) if adapter_fusion else None,
                    shared_adapter,
                    use_adapters,
                ),
            )
        else:
            raise NotImplementedError("Only support self-attention decoder layer")
        if use_output_layer:
            self.output_layer = torch.nn.ModuleDict()
            for lang in odim_dict.keys():
                self.output_layer[lang] = torch.nn.Linear(attention_dim, odim_dict[lang])
        else:
            self.output_layer = None
    def forward(self, tgt, tgt_mask, memory, memory_mask, language, use_adapter_fusion=True):
        x = self.embed[language](tgt)
        x, tgt_mask, memory, memory_mask, _, _, _ = self.decoders(
            x, tgt_mask, memory, memory_mask, language, None, use_adapter_fusion
        )
        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer[language](x)
        return x, tgt_mask
    def forward_one_step(self, tgt, tgt_mask, memory, memory_mask, language, cache=None):
        x = self.embed[language](tgt)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask, _, _, _ = decoder(
                x, tgt_mask, memory, None, language, cache=c,
            )
            new_cache.append(x)
        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer[language](y), dim=-1)
        return y, new_cache

class E2E(E2ETransformer):
    def __init__(self, idim, odim_dict, args, languages, ignore_id=-1):
        super().__init__(idim, 1, args, ignore_id)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        if args.fusion_languages:
            args.fusion_languages = args.fusion_languages.split("_")
        self.fusion_languages = sorted(args.fusion_languages) if args.fusion_languages else sorted(languages)
        self.encoder = AdaptiveEncoder(
            languages=languages,
            idim=idim,
            selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            conv_wshare=args.wshare,
            conv_kernel_length=args.ldconv_encoder_kernel_length,
            conv_usebias=args.ldconv_usebias,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            adapter_fusion=args.adapter_fusion,
            shared_adapter=args.shared_adapter,
            use_adapters=args.use_adapters,
            fusion_languages=self.fusion_languages
        )
        if args.mtlalpha < 1:
            self.decoder = AdaptiveDecoder(
                languages=languages,
                odim_dict=odim_dict,
                selfattention_layer_type=args.transformer_decoder_selfattn_layer_type,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                conv_wshare=args.wshare,
                conv_kernel_length=args.ldconv_decoder_kernel_length,
                conv_usebias=args.ldconv_usebias,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                src_attention_dropout_rate=args.transformer_attn_dropout_rate,
                adapter_fusion=args.adapter_fusion,
                shared_adapter=args.shared_adapter,
                use_adapters=args.use_adapters,
                fusion_languages=self.fusion_languages,
            )
        else:
            self.decoder = None
        self.soss = {lang: odim_dict[lang] - 1 for lang in languages}
        self.eoss = {lang: odim_dict[lang] - 1 for lang in languages}
        self.odim_dict = odim_dict

        self.criterion = torch.nn.ModuleDict()
        for lang in languages:
            self.criterion[lang] = LabelSmoothingLoss(
                                        self.odim_dict[lang],
                                        self.ignore_id,
                                        args.lsm_weight,
                                        args.transformer_length_normalized_loss,
                                    )
        if args.mtlalpha > 0.0:
            self.ctc = torch.nn.ModuleDict()
            for lang in languages:
                self.ctc[lang] = CTC(
                    odim_dict[lang], args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
                )
        else:
            self.ctc = None

        if args.report_cer or args.report_wer:
            self.error_calculator = ErrorCalculator(
                args.char_list,
                args.sym_space,
                args.sym_blank,
                args.report_cer,
                args.report_wer,
            )
        else:
            self.error_calculator = None

        self.reset_parameters(args)
        # Adapter
        self.meta_train = args.meta_train
        self.shared_adapter = args.shared_adapter
        self.adapter_fusion = False
        self.use_adapters = args.use_adapters
        if self.use_adapters:
            for p in self.parameters():
                p.requires_grad = False
            self.adapter_fusion = args.adapter_fusion
            if not args.adapter_fusion:
                adapter_train_languages = args.adapter_train_languages
                self.enable_adapter_training(adapter_train_languages, 
                            shared_adapter=args.shared_adapter, enable_head=args.train_adapter_with_head)
            else:
                self.reset_adapter_fusion_parameters()
                self.enable_adapter_fusion_training()
        self.recognize_language_branch = None # Set default recognize language for decoding

    def reset_adapter_fusion_parameters(self):
        key = "_".join(self.fusion_languages)
        for layer in self.encoder.encoders:
            layer.adapter_fusion[key].linear_v.weight.data = (
                torch.zeros(self.adim, self.adim) + 0.000001
            ).fill_diagonal_(1.0)
        for layer in self.decoder.decoders:
            layer.adapter_fusion[key].linear_v.weight.data = (
                torch.zeros(self.adim, self.adim) + 0.000001
            ).fill_diagonal_(1.0)
    def enable_adapter_fusion_training(self):
        key = "_".join(self.fusion_languages)
        logging.warning(f"Unfreezing the AdapterFusion parameters: {key}")
        for layer in self.encoder.encoders:
            for p in layer.adapter_fusion[key].parameters():
                p.requires_grad = True
            # for p in layer.adapter_norm.parameters():
            #     p.requires_grad = True
        for layer in self.decoder.decoders:
            for p in layer.adapter_fusion[key].parameters():
                p.requires_grad = True
    def get_fusion_guide_loss(self, language):
        device = next(self.parameters()).device
        if language not in self.fusion_languages:
            return torch.tensor(0.0).to(device)
        guide_loss = torch.tensor(0.0).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        
        lang_id = sorted(self.fusion_languages).index(language)
        key = "_".join(self.fusion_languages)
        target = torch.tensor(lang_id).unsqueeze(0).to(device)
        for layer in self.encoder.encoders:
            logits = layer.adapter_fusion[key].attn.mean(axis=(0, 1)).unsqueeze(0) # (batch, time, n_adapters)
            guide_loss = guide_loss + loss_fn(logits.exp(), target)
            layer.adapter_fusion.attn = None
        for layer in self.decoder.decoders:
            logits = layer.adapter_fusion[key].attn.mean(axis=(0, 1)).unsqueeze(0) # (batch, time, n_adapters)
            guide_loss = guide_loss + loss_fn(logits.exp(), target)
            layer.adapter_fusion[key].attn = None
        return guide_loss

    def get_fusion_regularization_loss(self):
        reg_loss = 0.0
        fusion_reg_loss_weight = 0.01
        device = next(self.parameters()).device
        key = "_".join(self.fusion_languages)
        target = torch.zeros((self.adim, self.adim)).fill_diagonal_(1.0).to(device)
        for layer in self.encoder.encoders:
            reg_loss = reg_loss + fusion_reg_loss_weight * (target - layer.adapter_fusion[key].linear_v.weight).pow(2).sum()
        for layer in self.decoder.decoders:
            reg_loss = reg_loss + fusion_reg_loss_weight * (target - layer.adapter_fusion[key].linear_v.weight).pow(2).sum()
        return reg_loss

    def enable_adapter_training(self, specified_languages=None, shared_adapter=False, enable_head=False):
        # Unfreeze the adapter parameters
        if specified_languages:
            enable_languages = specified_languages
        else:
            enable_languages = self.criterion.keys()
        logging.warning(f"Unfreezing the adapter parameters of {' '.join(enable_languages)}")
        for lang in enable_languages:
            if enable_head:
                for p in self.decoder.embed[lang].parameters():
                    p.requires_grad = True
                for p in self.decoder.output_layer[lang].parameters():
                    p.requires_grad = True
                for p in self.ctc[lang].parameters():
                    p.requires_grad = True
            if shared_adapter:
                lang = shared_adapter
            for layer in self.encoder.encoders:
                for p in layer.adapters[lang].parameters():
                    p.requires_grad = True
            for layer in self.decoder.decoders:
                for p in layer.adapters[lang].parameters():
                    p.requires_grad = True
    
    def forward(self, xs_pad, ilens, ys_pad, language, use_adapter_fusion=True):
        # 1. forward encoder
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask, language, use_adapter_fusion=True)
        self.hs_pad = hs_pad

        # 2. forward decoder
        if self.decoder is not None:
            ys_in_pad, ys_out_pad = add_sos_eos(
                ys_pad, self.soss[language], self.eoss[language], self.ignore_id
            )
            ys_mask = target_mask(ys_in_pad, self.ignore_id)
            pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask, language, use_adapter_fusion=True)
            self.pred_pad = pred_pad

            # 3. compute attention loss
            loss_att = self.criterion[language](pred_pad, ys_out_pad)
            self.acc = th_accuracy(
                pred_pad.view(-1, self.odim_dict[language]), ys_out_pad, ignore_label=self.ignore_id
            )
        else:
            loss_att = None
            self.acc = None

        # TODO(karita) show predicted text
        # TODO(karita) calculate these stats
        cer_ctc = None
        if self.mtlalpha == 0.0:
            loss_ctc = None
        else:
            batch_size = xs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            ctc_pred_pad = self.ctc[language].ctc_lo(hs_pad)
            loss_ctc = self.ctc[language](hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)
            if not self.training and self.error_calculator is not None:
                ys_hat = self.ctc[language].argmax(hs_pad.view(batch_size, -1, self.adim)).data
                cer_ctc = self.error_calculator[language](ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
            # for visualization
            if not self.training:
                self.ctc[language].softmax(hs_pad)
        
        # 5. compute cer/wer
        if self.training or self.error_calculator is None or self.decoder is None:
            cer, wer = None, None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        # copied from e2e_asr
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = loss_ctc
            loss_att_data = None
            loss_ctc_data = float(loss_ctc)
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = float(loss_ctc)
        
        if self.adapter_fusion and self.training:
            guide_loss = self.get_fusion_guide_loss(language)
        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(
                loss_ctc_data, loss_att_data, self.acc, cer_ctc, cer, wer, loss_data
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)
        if self.training and not self.meta_train:
            if not self.adapter_fusion:
                guide_loss = torch.tensor(0.0).cuda()
            return (self.loss, guide_loss)
        return self.loss
    
    def calculate_all_ctc_probs(self, xs_pad, ilens, ys_pad, language):
        """E2E CTC probability calculation.
        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: CTC probability (B, Tmax, vocab)
        :rtype: float ndarray
        """
        ret = None
        if self.mtlalpha == 0:
            return ret

        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad, language)
        for name, m in self.named_modules():
            if isinstance(m, CTC) and m.probs is not None:
                ret = m.probs.cpu().numpy()
        self.train()
        return ret
    
    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, language):
        """E2E attention calculation.
        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights (B, H, Lmax, Tmax)
        :rtype: float ndarray
        """
        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad, language)
        ret = dict()
        for name, m in self.named_modules():
            if (
                isinstance(m, MultiHeadedAttention)
                or isinstance(m, DynamicConvolution)
                or isinstance(m, RelPositionMultiHeadedAttention)
            ):
                ret[name] = m.attn.cpu().numpy()
            if isinstance(m, DynamicConvolution2D):
                ret[name + "_time"] = m.attn_t.cpu().numpy()
                ret[name + "_freq"] = m.attn_f.cpu().numpy()
        self.train()
        return ret
    
    def calculate_adapter_fusion_attentions(self, xs_pad, ilens, ys_pad, language):
        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad, language)
        ret = dict()
        for name, m in self.named_modules():
            if (
                isinstance(m, MultiHeadedAttention)
                or isinstance(m, DynamicConvolution)
                or isinstance(m, RelPositionMultiHeadedAttention)
            ) and "adapter_fusion" in name:
                ret[name] = m.attn.cpu().numpy()
        return ret

    def encode(self, x, language):
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)
        enc_output, _ = self.encoder(x, None, language)
        return enc_output.squeeze(0)
    
    def set_recognize_language_branch(self, language):
        self.recognize_language_branch = language
    def recognize_batch(self, xs, recog_args, char_list=None, rnnlm=None, language=None):
        assert language is not None or self.recognize_language_branch is not None, \
                                "Recognize language is not specified"
        if language is None:
            language = self.recognize_language_branch
        prev = self.training
        self.eval()
        ilens = numpy.fromiter((xx.shape[0] for xx in xs), dtype=numpy.int64)
        # subsample frame
        xs = [xx[:: self.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]

        xs_pad = pad_list(xs, 0.0)

        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)

        # 1. Encoder
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask, language)
        hlens = torch.tensor([int(sum(mask[0])) for mask in hs_mask])

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc[language].log_softmax(hs_pad)
            normalize_score = False
        else:
            lpz = None
            normalize_score = True

        logging.info("max input length: " + str(hs_pad.size(1)))

        # search params
        batch = len(hlens)
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = getattr(recog_args, "ctc_weight", 0)  # for NMT
        att_weight = 1.0 - ctc_weight

        n_bb = batch * beam
        pad_b = to_device(hs_pad, torch.arange(batch) * beam).view(-1, 1)
        max_hlens = hlens

        if recog_args.maxlenratio == 0:
            maxlens = max_hlens
        else:
            maxlens = [
                max(1, int(recog_args.maxlenratio * max_hlen)) for max_hlen in max_hlens
            ]
        minlen = min([int(recog_args.minlenratio * max_hlen) for max_hlen in max_hlens])
        logging.info("max output lengths: " + str(maxlens))
        logging.info("min output length: " + str(minlen))

        vscores = to_device(hs_pad, torch.zeros(batch, beam))
        rnnlm_state = None

        import six

        # initialize hypothesis
        yseq = [[self.soss[language]] for _ in six.moves.range(n_bb)]
        accum_odim_ids = [self.soss[language] for _ in six.moves.range(n_bb)]
        stop_search = [False for _ in six.moves.range(batch)]
        nbest_hyps = [[] for _ in six.moves.range(batch)]
        ended_hyps = [[] for _ in six.moves.range(batch)]

        exp_hs_mask = (
            hs_mask.unsqueeze(1).repeat(1, beam, 1, 1).contiguous()
        )  # (batch, beam, 1, T)
        exp_hs_mask = exp_hs_mask.view(n_bb, hs_mask.size()[1], hs_mask.size()[2])
        exp_h = (
            hs_pad.unsqueeze(1).repeat(1, beam, 1, 1).contiguous()
        )  # (batch, beam, T, F)
        exp_h = exp_h.view(n_bb, hs_pad.size()[1], hs_pad.size()[2])

        ctc_scorer, ctc_state = None, None
        if lpz is not None:
            scoring_num = min(
                int(beam * CTC_SCORING_RATIO) if att_weight > 0.0 else 0,
                lpz.size(-1),
            )
            ctc_scorer = CTCPrefixScoreTH(lpz, hlens, 0, self.eoss[language])

        for i in six.moves.range(max(maxlens)):
            logging.debug("position " + str(i))

            # get nbest local scores and their ids
            ys_mask = subsequent_mask(i + 1).to(hs_pad.device).unsqueeze(0)

            ys = torch.tensor(yseq).to(hs_pad.device)
            vy = to_device(hs_pad, torch.LongTensor(self._get_last_yseq(yseq)))

            # local_att_scores (n_bb = beam * batch, vocab)
            if self.decoder is not None:
                local_att_scores = self.decoder.forward_one_step(
                    ys, ys_mask, exp_h, memory_mask=exp_hs_mask, language=language,
                )[0]
            else:
                local_att_scores = to_device(
                    hs_pad, torch.zeros((n_bb, lpz.size(-1)), dtype=lpz.dtype)
                )

            if rnnlm:
                rnnlm_state, local_lm_scores = rnnlm.buff_predict(rnnlm_state, vy, n_bb)
                local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
            else:
                local_scores = local_att_scores

            # ctc
            if ctc_scorer:
                local_scores = att_weight * local_att_scores
                local_scores[:, 0] = self.logzero  # avoid choosing blank
                part_ids = (
                    torch.topk(local_scores, scoring_num, dim=-1)[1]
                    if scoring_num > 0
                    else None
                )
                local_ctc_scores, ctc_state = ctc_scorer(
                    yseq, ctc_state, part_ids
                )  # local_ctc_scores (n_bb, odim)

                local_scores = local_scores + ctc_weight * local_ctc_scores
                if rnnlm:
                    local_scores = local_scores + recog_args.lm_weight * local_lm_scores
            
            local_scores = local_scores.view(batch, beam, self.odim_dict[language])
            if i == 0:
                local_scores[:, 1:, :] = self.logzero

            # accumulate scores
            eos_vscores = local_scores[:, :, self.eoss[language]] + vscores
            vscores = vscores.view(batch, beam, 1).repeat(1, 1, self.odim_dict[language])
            vscores[:, :, self.eoss[language]] = self.logzero
            vscores = (vscores + local_scores).view(batch, -1)  # (batch, odim * beam)

            # global pruning
            accum_best_scores, accum_best_ids = torch.topk(vscores, beam, 1)

            accum_odim_ids = (
                torch.fmod(accum_best_ids, self.odim_dict[language]).view(-1).data.cpu().tolist()
            )
            accum_padded_beam_ids = (
                (accum_best_ids // self.odim_dict[language] + pad_b).view(-1).data.cpu().tolist()
            )
            y_prev = yseq[:][:]
            yseq = self._index_select_list(yseq, accum_padded_beam_ids)
            yseq = self._append_ids(yseq, accum_odim_ids)
            vscores = accum_best_scores

            vidx = to_device(hs_pad, torch.LongTensor(accum_padded_beam_ids))

            # pick ended hyps
            if i >= minlen:
                k = 0
                penalty_i = (i + 1) * penalty
                thr = accum_best_scores[:, -1]
                for samp_i in six.moves.range(batch):
                    if stop_search[samp_i]:
                        k = k + beam
                        continue
                    for beam_j in six.moves.range(beam):
                        _vscore = None
                        if eos_vscores[samp_i, beam_j] > thr[samp_i]:
                            yk = y_prev[k][:]
                            if len(yk) <= maxlens[samp_i]:
                                _vscore = eos_vscores[samp_i][beam_j] + penalty_i
                            rnnlm_idx = k
                        elif i == maxlens[samp_i] - 1:
                            yk = yseq[k][:]
                            _vscore = vscores[samp_i][beam_j] + penalty_i
                            rnnlm_idx = accum_padded_beam_ids[k]
                        if _vscore:
                            yk.append(self.eoss[language])
                            if rnnlm:
                                _vscore += recog_args.lm_weight * rnnlm.final(
                                    rnnlm_state, index=rnnlm_idx
                                )
                            ended_hyps[samp_i].append(
                                {"yseq": yk, "score": _vscore.data.cpu().numpy()}
                            )
                        k = k + 1
            # end detection
            stop_search = [
                stop_search[samp_i]
                or end_detect(ended_hyps[samp_i], i)
                or i >= maxlens[samp_i]
                for samp_i in six.moves.range(batch)
            ]

            stop_search_summary = list(set(stop_search))

            if len(stop_search_summary) == 1 and stop_search_summary[0]:
                break

            if rnnlm:
                rnnlm_state = self._index_select_lm_state(rnnlm_state, 0, vidx)
            if ctc_scorer:
                ctc_state = ctc_scorer.index_select_state(ctc_state, accum_best_ids)

        torch.cuda.empty_cache()

        dummy_hyps = [
            {"yseq": [self.soss[language], self.eoss[language]], "score": numpy.array([-float("inf")])}
        ]
        ended_hyps = [
            ended_hyps[samp_i] if len(ended_hyps[samp_i]) != 0 else dummy_hyps
            for samp_i in six.moves.range(batch)
        ]
        if normalize_score:
            for samp_i in six.moves.range(batch):
                for x in ended_hyps[samp_i]:
                    x["score"] /= len(x["yseq"])

        nbest_hyps = [
            sorted(ended_hyps[samp_i], key=lambda x: x["score"], reverse=True)[
                : min(len(ended_hyps[samp_i]), recog_args.nbest)
            ]
            for samp_i in six.moves.range(batch)
        ]
        if prev:
            self.train()
        return nbest_hyps

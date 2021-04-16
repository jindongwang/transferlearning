import collections
from espnet.nets.pytorch_backend.e2e_asr_transformer import *
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E as SpeechTransformer
from espnet.nets.pytorch_backend.transformer.encoder import *
from espnet.nets.pytorch_backend.transformer.decoder import *
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.decoder_layer import DecoderLayer
import torch
from distances import CORAL, MMD_loss
import numpy as np
from ctc_aligner import CTCForcedAligner

def adapt_loss(source, target, adapt_loss="mmd"):
    if adapt_loss == "mmd": # 1.0 level
        mmd_loss = MMD_loss()
        loss = mmd_loss(source, target)
    elif adapt_loss == "mmd_linear":
        mmd_loss = MMD_loss(kernel_type="linear")
        loss = mmd_loss(source, target)
    elif adapt_loss == "coral": # 1e-4 level
        loss = CORAL(source, target)
    else:
        raise NotImplementedError(f"Adapt loss type {adapt_loss} is not implemented")
    return loss

class Discriminator(torch.nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dis1 = torch.nn.Linear(input_dim, hidden_dim)
        self.bn = torch.nn.BatchNorm1d(hidden_dim)
        self.dis2 = torch.nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = torch.nn.functional.relu(self.dis1(x))
        x = self.dis2(self.bn(x.permute(0, 2, 1)).permute(0, 2, 1))
        x = torch.sigmoid(x)
        return x

class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class CustomEncoderLayer(EncoderLayer):
    def forward(self, x, mask, cache=None):
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
            self.x_norm = x
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
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)
        return x, mask

class CustomEncoder(Encoder):
    def __init__(
        self,
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
        encoder_selfattn_layer = MultiHeadedAttention
        encoder_selfattn_layer_args = [
            (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        ] * num_blocks
        self.encoders = repeat(
            num_blocks,
            lambda lnum: CustomEncoderLayer(
                attention_dim,
                encoder_selfattn_layer(*encoder_selfattn_layer_args[lnum]),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
    def forward(self, xs, masks, return_repr=False):
        """Encode input sequence.
        Args:
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, time).
        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, time).
        """
        xs, masks = self.embed(xs, masks)
        #xs, masks = self.encoders(xs, masks)
        final_repr = []
        for layer_idx, e in enumerate(self.encoders):
            xs, masks = e(xs, masks)
            if return_repr and layer_idx > 0:
                assert e.x_norm is not None
                final_repr.append(e.x_norm)
                #print(e.x_norm.mean(), xs.mean())
                e.x_norm = None
        if self.normalize_before:
            xs = self.after_norm(xs)
            final_repr.append(xs)
        return (xs, masks) if not return_repr else (xs, masks, final_repr)

class CustomDecoderLayer(DecoderLayer):
    def forward(self, tgt, tgt_mask, memory, memory_mask, cache=None):
        """Compute decoded features.
        Args:
            tgt (torch.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (torch.Tensor): Mask for input tensor (#batch, maxlen_out).
            memory (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in, size).
            memory_mask (torch.Tensor): Encoded memory mask (#batch, maxlen_in).
            cache (List[torch.Tensor]): List of cached tensors.
                Each tensor shape should be (#batch, maxlen_out - 1, size).
        Returns:
            torch.Tensor: Output tensor(#batch, maxlen_out, size).
            torch.Tensor: Mask for output tensor (#batch, maxlen_out).
            torch.Tensor: Encoded memory (#batch, maxlen_in, size).
            torch.Tensor: Encoded memory mask (#batch, maxlen_in).
        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
            self.x_norm = tgt
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
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, tgt_mask, memory, memory_mask
class CustomDecoder(Decoder):
    def __init__(
        self,
        odim,
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
    ):
        super().__init__(odim,
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
            concat_after,
        )
        decoder_selfattn_layer = MultiHeadedAttention
        decoder_selfattn_layer_args = [
            (
                attention_heads,
                attention_dim,
                self_attention_dropout_rate,
            )
        ] * num_blocks
        self.decoders = repeat(
                num_blocks,
                lambda lnum: CustomDecoderLayer(
                    attention_dim,
                    decoder_selfattn_layer(*decoder_selfattn_layer_args[lnum]),
                    MultiHeadedAttention(
                        attention_heads, attention_dim, src_attention_dropout_rate
                    ),
                    PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )

    def forward(self, tgt, tgt_mask, memory, memory_mask, return_repr=False):
        """Forward decoder.
        Args:
            tgt (torch.Tensor): Input token ids, int64 (#batch, maxlen_out) if
                input_layer == "embed". In the other case, input tensor
                (#batch, maxlen_out, odim).
            tgt_mask (torch.Tensor): Input token mask (#batch, maxlen_out).
                dtype=torch.uint8 in PyTorch 1.2- and dtype=torch.bool in PyTorch 1.2+
                (include 1.2).
            memory (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in, feat).
            memory_mask (torch.Tensor): Encoded memory mask (#batch, maxlen_in).
                dtype=torch.uint8 in PyTorch 1.2- and dtype=torch.bool in PyTorch 1.2+
                (include 1.2).
        Returns:
            torch.Tensor: Decoded token score before softmax (#batch, maxlen_out, odim)
                   if use_output_layer is True. In the other case,final block outputs
                   (#batch, maxlen_out, attention_dim).
            torch.Tensor: Score mask before softmax (#batch, maxlen_out).
        """
        x = self.embed(tgt)
        # x, tgt_mask, memory, memory_mask = self.decoders(
        #     x, tgt_mask, memory, memory_mask
        # )
        final_repr = []
        for layer_idx, decoder in enumerate(self.decoders):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, memory_mask
            )
            if return_repr and layer_idx > 0:
                assert decoder.x_norm is not None
                final_repr.append(decoder.x_norm)
                decoder.x_norm = None
        if self.normalize_before:
            x = self.after_norm(x)
            final_repr.append(x)
        if self.output_layer is not None:
            x = self.output_layer(x)
        return (x, tgt_mask, None) if not return_repr else (x, tgt_mask, final_repr)
        
class CustomSpeechTransformer(SpeechTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        idim, odim = args
        args = kwargs["args"]
        self.encoder = CustomEncoder(
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
        )
        self.decoder = CustomDecoder(
            odim=odim,
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
        )
class UDASpeechTransformer(CustomSpeechTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args = kwargs["args"]
        assert hasattr(args, "transfer_type")
        assert isinstance(args.self_training, bool)
        self.loss_type = args.transfer_type
        self.self_training = args.self_training
        self.n_classes = len(args.char_list)

        self.multi_enc_repr_num = args.multi_enc_repr_num
        self.multi_dec_repr_num = args.multi_dec_repr_num
        self.use_dec_repr = args.use_dec_repr
        self.pseudo_ctc_confidence_thr = args.pseudo_ctc_confidence_thr # Threshold for filtering CTC outputs
        self.cmatch_method = args.cmatch_method
        self.ctc_aligner = CTCForcedAligner(char_list=None)
        self.char_list = args.char_list
        self.bpemodel = None
        self.non_char_symbols = list(map(int, args.non_char_symbols.split("_")))
        if self.loss_type:
            if "cmatch" in self.loss_type:
                assert args.cmatch_method is not None, "CMatch method is required."
                assert self.non_char_symbols is not None, "Non-character symbol list must be specified"
            elif self.loss_type == "adv":
                self.domain_classifier = Discriminator(input_dim=self.adim, hidden_dim=self.adim)
    
    def forward(self, 
                xs_pad, 
                ilens, 
                ys_pad, 
                tgt_xs_pad=None, 
                tgt_ilens=None, 
                tgt_ys_pad=None):
        """E2E forward.
        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        dec_return_repr = True
        enc_return_repr = True

        # 1. forward encoder
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        if enc_return_repr:
            hs_pad, hs_mask, src_enc_repr = self.encoder(xs_pad, src_mask, return_repr=enc_return_repr)
            src_enc_repr = src_enc_repr[-self.multi_enc_repr_num:]
        else:
            hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        self.hs_pad = hs_pad

        # 2. forward decoder
        if self.decoder is not None:
            ys_in_pad, ys_out_pad = add_sos_eos(
                ys_pad, self.sos, self.eos, self.ignore_id
            )
            ys_mask = target_mask(ys_in_pad, self.ignore_id)
            pred_pad, pred_mask, src_dec_repr = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask, return_repr=dec_return_repr)

            if src_dec_repr:
                src_dec_repr = src_dec_repr[-self.multi_dec_repr_num:]
            self.pred_pad = pred_pad

            # 3. compute attention loss
            loss_att = self.criterion(pred_pad, ys_out_pad)
            self.acc = th_accuracy(
                pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
            )
        else:
            loss_att = None
            self.acc = None

        cer_ctc = None
        if self.mtlalpha == 0.0:
            loss_ctc = None
        else:
            batch_size = xs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)
            if not self.training and self.error_calculator is not None:
                ys_hat = self.ctc.argmax(hs_pad.view(batch_size, -1, self.adim)).data
                cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
            # for visualization
            if not self.training:
                self.ctc.softmax(hs_pad)

        # 5. compute cer/wer
        if self.training or self.error_calculator is None or self.decoder is None:
            cer, wer = None, None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())
        
        src_hlens = torch.tensor([int(sum(mask[0])) for mask in hs_mask])
        if self.cmatch_method == "pseudo_ctc_pred":
            src_hs_flatten = torch.cat([hs_pad[i, :src_hlens[i], :].view(-1, self.adim) for i in range(len(hs_pad))]) # hs_pad: B * T, F
            src_ctc_softmax = torch.nn.functional.softmax(self.ctc.ctc_lo(src_hs_flatten), dim=1)
        else:
            src_ctc_softmax = None

        # Domain adversarial loss
        if tgt_xs_pad is not None and tgt_ilens is not None:
            src_ys_pad = ys_pad
            src_hs_pad, src_hs_mask = hs_pad, hs_mask
            tgt_xs_pad = tgt_xs_pad[:, : max(tgt_ilens)]  # for data parallel
            tgt_src_mask = make_non_pad_mask(tgt_ilens.tolist()).to(tgt_xs_pad.device).unsqueeze(-2)
            
            if enc_return_repr:
                tgt_hs_pad, tgt_hs_mask, tgt_enc_repr = self.encoder(tgt_xs_pad, tgt_src_mask, return_repr=enc_return_repr)
                tgt_enc_repr = tgt_enc_repr[-self.multi_enc_repr_num:]
            else:
                tgt_hs_pad, tgt_hs_mask = self.encoder(tgt_xs_pad, tgt_src_mask)

            src_ys_out_pad = ys_out_pad
            src_ys_out_flatten = src_ys_out_pad.contiguous().view(-1)

            if tgt_ys_pad is not None:
                tgt_ys_in_pad, tgt_ys_out_pad = add_sos_eos(
                    tgt_ys_pad, self.sos, self.eos, self.ignore_id
                )
                tgt_ys_mask = target_mask(tgt_ys_in_pad, self.ignore_id)
                tgt_pred_pad, tgt_pred_mask, tgt_dec_repr = self.decoder(tgt_ys_in_pad, tgt_ys_mask, tgt_hs_pad, tgt_hs_mask, return_repr=dec_return_repr)
                if tgt_dec_repr:
                    tgt_dec_repr = tgt_dec_repr[-self.multi_dec_repr_num:]
                tgt_ys_out_flatten = tgt_ys_out_pad.contiguous().view(-1)
            
            tgt_hlens = torch.tensor([int(sum(mask[0])) for mask in tgt_hs_mask])
            if self.cmatch_method == "pseudo_ctc_pred":
                tgt_hs_flatten = torch.cat([tgt_hs_pad[i, :tgt_hlens[i], :].view(-1, self.adim) for i in range(len(tgt_hs_pad))]) # hs_pad: B * T, F
                tgt_ctc_softmax = torch.nn.functional.softmax(self.ctc.ctc_lo(tgt_hs_flatten), dim=1)
            else:
                tgt_ctc_softmax = None

            if self.self_training:
                src_loss_att = loss_att
                src_loss_ctc = loss_ctc
                tgt_batch_size = tgt_xs_pad.size(0)
                tgt_hs_len = tgt_hs_mask.view(tgt_batch_size, -1).sum(1)
                tgt_loss_ctc = self.ctc(tgt_hs_pad.view(tgt_batch_size, -1, self.adim), tgt_hs_len, tgt_ys_pad)
                tgt_loss_att = self.criterion(tgt_pred_pad, tgt_ys_out_pad)

                loss_att = (src_loss_att + tgt_loss_att) / 2
                loss_ctc = (src_loss_ctc + tgt_loss_ctc) / 2
                self.acc = (self.acc + th_accuracy(
                    tgt_pred_pad.view(-1, self.odim), tgt_ys_out_pad, ignore_label=self.ignore_id
                )) / 2
            
            uda_loss = torch.tensor(0.0).cuda()
            if not self.loss_type:
                uda_loss = torch.tensor(0.0).cuda()
            elif self.loss_type == "adv":
                uda_loss = self.adversarial_loss(src_hs_pad, tgt_hs_pad)
            elif self.loss_type == "cmatch":
                assert tgt_ys_pad is not None
                assert len(src_enc_repr) == self.multi_enc_repr_num and len(src_enc_repr) in [1, 3, 6, 9, 12], len(src_enc_repr)
                for layer_idx in range(len(src_enc_repr)):
                    src_hs_flatten, src_ys_flatten, tgt_hs_flatten, tgt_ys_flatten \
                                = self.get_enc_repr(src_enc_repr[layer_idx], 
                                                    src_hlens, 
                                                    tgt_enc_repr[layer_idx], 
                                                    tgt_hlens,
                                                    src_ys_pad,
                                                    tgt_ys_pad,
                                                    method=self.cmatch_method,
                                                    src_ctc_softmax=src_ctc_softmax,
                                                    tgt_ctc_softmax=tgt_ctc_softmax,)
                    layer_uda_loss = self.cmatch_loss_func(self.n_classes,
                                            src_hs_flatten, 
                                            src_ys_flatten, 
                                            tgt_hs_flatten, 
                                            tgt_ys_flatten)
                    uda_loss = uda_loss + layer_uda_loss if uda_loss else layer_uda_loss
                if self.use_dec_repr:
                    assert len(src_dec_repr) == self.multi_dec_repr_num, len(src_dec_repr)
                    # No need to calculate decoder matching loss
                    for layer_idx in range(len(src_dec_repr)):
                        src_repr_flatten = src_dec_repr[layer_idx].contiguous().view(-1, self.adim)
                        tgt_repr_flatten = tgt_dec_repr[layer_idx].contiguous().view(-1, self.adim)
                        layer_uda_loss = self.cmatch_loss_func(self.n_classes,
                                                        src_repr_flatten, 
                                                        src_ys_out_flatten, 
                                                        tgt_repr_flatten, 
                                                        tgt_ys_out_flatten)
                        uda_loss = uda_loss + layer_uda_loss
            elif self.loss_type in ["coral", "mmd"]:
                # (B, T, F) --> (B, F)
                uda_loss = adapt_loss(torch.mean(src_hs_pad, dim=1), 
                                    torch.mean(tgt_hs_pad, dim=1), 
                                    adapt_loss=self.loss_type)
            else:
                raise NotImplementedError(f"loss type {self.loss_type} is not implemented")

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

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(
                loss_ctc_data, loss_att_data, self.acc, cer_ctc, cer, wer, loss_data
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)
        return (self.loss, uda_loss) if (self.training and uda_loss is not None) else self.loss
    
    def adversarial_loss(self, src_hs_pad, tgt_hs_pad, alpha=1.0):
        loss_fn = torch.nn.BCELoss()
        src_hs_pad = ReverseLayerF.apply(src_hs_pad, alpha)
        
        tgt_hs_pad = ReverseLayerF.apply(tgt_hs_pad, alpha)
        src_domain = self.domain_classifier(src_hs_pad).view(-1, 1) # B, T, 1
        tgt_domain = self.domain_classifier(tgt_hs_pad).view(-1, 1) # B, T, 1
        device = src_hs_pad.device
        src_label = torch.ones(len(src_domain)).long().to(device)
        tgt_label = torch.zeros(len(tgt_domain)).long().to(device)
        domain_pred = torch.cat([src_domain, tgt_domain], dim=0)
        domain_label = torch.cat([src_label, tgt_label], dim=0)
        uda_loss = loss_fn(domain_pred, domain_label[:, None].float()) # B, 1
        return uda_loss

    def get_enc_repr(self, 
                    src_hs_pad, 
                    src_hlens, 
                    tgt_hs_pad, 
                    tgt_hlens, 
                    src_ys_pad, 
                    tgt_ys_pad, 
                    method,
                    src_ctc_softmax=None,
                    tgt_ctc_softmax=None):
        src_ys = [y[y != self.ignore_id] for y in src_ys_pad]
        tgt_ys = [y[y != self.ignore_id] for y in tgt_ys_pad]
        if method == "frame_average":
            def frame_average(hidden_states, num):
                # hs_i, B T F
                hidden_states = hidden_states.permute(0, 2, 1)
                downsampled_states = torch.nn.functional.adaptive_avg_pool1d(hidden_states, num)
                downsampled_states = downsampled_states.permute(0, 2, 1)
                assert downsampled_states.shape[1] == num, f"{downsampled_states.shape[1]}, {num}"
                return downsampled_states
            src_hs_downsampled = frame_average(src_hs_pad, num=src_ys_pad.size(1))
            tgt_hs_downsampled = frame_average(tgt_hs_pad, num=tgt_ys_pad.size(1))
            src_hs_flatten = src_hs_downsampled.contiguous().view(-1, self.adim)
            tgt_hs_flatten = tgt_hs_downsampled.contiguous().view(-1, self.adim)
            src_ys_flatten = src_ys_pad.contiguous().view(-1)
            tgt_ys_flatten = tgt_ys_pad.contiguous().view(-1)
        elif method == "ctc_align":
            src_ys = [y[y != -1] for y in src_ys_pad]
            src_logits = self.ctc.ctc_lo(src_hs_pad)
            src_align_pad = self.ctc_aligner(src_logits, src_hlens, src_ys)
            src_ys_flatten = torch.cat([src_align_pad[i, :src_hlens[i]].view(-1) for i in range(len(src_align_pad))])
            src_hs_flatten = torch.cat([src_hs_pad[i, :src_hlens[i], :].view(-1, self.adim) for i in range(len(src_hs_pad))]) # hs_pad: B, T, F
            tgt_ys = [y[y != -1] for y in tgt_ys_pad]
            tgt_logits = self.ctc.ctc_lo(tgt_hs_pad)
            tgt_align_pad = self.ctc_aligner(tgt_logits, tgt_hlens, tgt_ys)
            tgt_ys_flatten = torch.cat([tgt_align_pad[i, :tgt_hlens[i]].view(-1) for i in range(len(tgt_align_pad))])
            tgt_hs_flatten = torch.cat([tgt_hs_pad[i, :tgt_hlens[i], :].view(-1, self.adim) for i in range(len(tgt_hs_pad))]) # hs_pad: B, T, F
        elif method == "pseudo_ctc_pred":
            assert src_ctc_softmax is not None
            src_hs_flatten = torch.cat([src_hs_pad[i, :src_hlens[i], :].view(-1, self.adim) for i in range(len(src_hs_pad))]) # hs_pad: B * T, F
            src_hs_flatten_size = src_hs_flatten.shape[0]
            src_confidence, src_ctc_ys = torch.max(src_ctc_softmax, dim=1)
            src_confidence_mask = (src_confidence > self.pseudo_ctc_confidence_thr)
            src_ys_flatten = src_ctc_ys[src_confidence_mask]
            src_hs_flatten = src_hs_flatten[src_confidence_mask]

            assert tgt_ctc_softmax is not None
            tgt_hs_flatten = torch.cat([tgt_hs_pad[i, :tgt_hlens[i], :].view(-1, self.adim) for i in range(len(tgt_hs_pad))]) # hs_pad: B * T, F
            tgt_hs_flatten_size = tgt_hs_flatten.shape[0]
            tgt_confidence, tgt_ctc_ys = torch.max(tgt_ctc_softmax, dim=1)
            tgt_confidence_mask = (tgt_confidence > self.pseudo_ctc_confidence_thr)
            
            tgt_ys_flatten = tgt_ctc_ys[tgt_confidence_mask]
            tgt_hs_flatten = tgt_hs_flatten[tgt_confidence_mask]
            # logging.warning(f"Source pseudo CTC ratio: {src_hs_flatten.shape[0] / src_hs_flatten_size:.2f}; " \
            #          f"Target pseudo CTC ratio: {tgt_hs_flatten.shape[0] / tgt_hs_flatten_size:.2f}")
        return src_hs_flatten, src_ys_flatten, tgt_hs_flatten, tgt_ys_flatten
    
    def cmatch_loss_func(self, n_classes, 
                            src_features, src_labels, 
                            tgt_features, tgt_labels):
        assert src_features.shape[0] == src_labels.shape[0]
        assert tgt_features.shape[0] == tgt_labels.shape[0]
        classes = torch.arange(n_classes)
        def src_token_idxs(c):
            return src_labels.eq(c).nonzero().squeeze(1)
        src_token_idxs = list(map(src_token_idxs, classes))
        def tgt_token_idxs(c):
            return tgt_labels.eq(c).nonzero().squeeze(1)
        tgt_token_idxs = list(map(tgt_token_idxs, classes))
        assert len(src_token_idxs) == n_classes
        assert len(tgt_token_idxs) == n_classes
        loss = torch.tensor(0.0).cuda()
        count = 0
        for c in classes:
            if c in self.non_char_symbols or src_token_idxs[c].shape[0] < 5 or tgt_token_idxs[c].shape[0] < 5:
                continue
            loss = loss + adapt_loss(src_features[src_token_idxs[c]], 
                                      tgt_features[tgt_token_idxs[c]], 
                                        adapt_loss='mmd_linear')
            count = count + 1
        loss = loss / count if count > 0 else loss
        return loss
import torch
import torch.nn as nn

from tst.encoder import Encoder
from tst.decoder import Decoder
from tst.utils import generate_original_PE, generate_regular_PE
from base.loss_transfer import TransferLoss

class Transformer(nn.Module):
    """Transformer model from Attention is All You Need.

    A classic transformer model adapted for sequential data.
    Embedding has been replaced with a fully connected layer,
    the last layer softmax is now a sigmoid.

    Attributes
    ----------
    layers_encoding: :py:class:`list` of :class:`Encoder.Encoder`
        stack of Encoder layers.
    layers_decoding: :py:class:`list` of :class:`Decoder.Decoder`
        stack of Decoder layers.

    Parameters
    ----------
    d_input:
        Model input dimension.
    d_model:
        Dimension of the input vector.
    d_output:
        Model output dimension.
    q:
        Dimension of queries and keys.
    v:
        Dimension of values.
    h:
        Number of heads.
    N:
        Number of encoder and decoder layers to stack.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    dropout:
        Dropout probability after each MHA or PFF block.
        Default is ``0.3``.
    chunk_mode:
        Switch between different MultiHeadAttention blocks.
        One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``.
    pe:
        Type of positional encoding to add.
        Must be one of ``'original'``, ``'regular'`` or ``None``. Default is ``None``.
    pe_period:
        If using the ``'regular'` pe, then we can define the period. Default is ``24``.
    """

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk',
                 pe: str = None,
                 pe_period: int = 24):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = d_model

        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
        self.layers_decoding = nn.ModuleList([Decoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])

        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)

        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')

        self.name = 'transformer'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate input through transformer

        Forward input through an embedding module,
        the encoder then decoder stacks, and an output module.

        Parameters
        ----------
        x:
            :class:`torch.Tensor` of shape (batch_size, K, d_input).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_output).
        """
        K = x.shape[1]

        # Embeddin module
        encoding = self._embedding(x)

        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)

        # Encoding stack
        list_encoding = []
        for layer in self.layers_encoding:
            encoding = layer(encoding)
            list_encoding.append(encoding)

        # Decoding stack
        decoding = encoding

        # Add position encoding
        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(K, self._d_model)
            positional_encoding = positional_encoding.to(decoding.device)
            decoding.add_(positional_encoding)

        for layer in self.layers_decoding:
            decoding = layer(decoding, encoding)

        # Output module
        output = self._linear(decoding)
        output = torch.sigmoid(output)
        return output, list_encoding

    # def adapt_encoding(self, list_encoding, loss_type, train_type):
    #     ## train_type "last" : last hidden, "all": all hidden
    #     loss_all = torch.zeros(1).cuda()
    #     for i in range(len(list_encoding)):
    #         data = list_encoding[i]
    #         data_s = data[0:len(data)//2]
    #         data_t = data[len(data)//2:]
    #         criterion_transder = TransferLoss(
    #             loss_type=loss_type, input_dim=data_s.shape[2])
    #         if train_type == 'last':
    #             loss_all = loss_all + criterion_transder.compute(
    #                     data_s[:, -1, :], data_t[:, -1, :])
    #         elif train_type == "all":
    #             for j in range(data_s.shape[1]):
    #                 loss_all = loss_all + criterion_transder.compute(data_s[:, j, :], data_t[:, j, :])
    #         else:
    #             print("adapt loss error!")
    #     return loss_all


    def adapt_encoding_weight(self, list_encoding, loss_type, train_type, weight_mat=None):
        loss_all = torch.zeros(1).cuda()
        len_seq = list_encoding[0].shape[1]
        num_layers = len(list_encoding)
        if weight_mat is None:
            weight = (1.0 / len_seq *
                      torch.ones(num_layers, len_seq)).cuda()
        else:
            weight = weight_mat
        dist_mat = torch.zeros(num_layers, len_seq).cuda()
        for i in range(len(list_encoding)):
            data = list_encoding[i]
            data_s = data[0:len(data)//2]
            data_t = data[len(data)//2:]
            criterion_transder = TransferLoss(
                loss_type=loss_type, input_dim=data_s.shape[2])
            if train_type == 'last':
                loss_all = loss_all + criterion_transder.compute(
                        data_s[:, -1, :], data_t[:, -1, :])
            elif train_type == "all":
                for j in range(data_s.shape[1]):
                    loss_transfer = criterion_transder.compute(data_s[:, j, :], data_t[:, j, :])
                    loss_all = loss_all + weight[i, j] * loss_transfer
                    dist_mat[i, j] = loss_transfer 
            else:
                print("adapt loss error!")
        return loss_all, dist_mat, weight

    def update_weight_Boosting(self, weight_mat, dist_old, dist_new):
        epsilon = 1e-12
        dist_old = dist_old.detach()
        dist_new = dist_new.detach()
        ind = dist_new > dist_old + epsilon
        weight_mat[ind] = weight_mat[ind] * \
            (1 + torch.sigmoid(dist_new[ind] - dist_old[ind]))
        weight_norm = torch.norm(weight_mat, dim=1, p=1)
        weight_mat = weight_mat / weight_norm.t().unsqueeze(1).repeat(1, len(weight_mat[0]))
        return weight_mat



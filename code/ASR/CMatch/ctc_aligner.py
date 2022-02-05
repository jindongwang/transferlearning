import torch
import numpy as np

'''
Borrowed and modified from neural_sp: 
https://github.com/hirofumi0810/neural_sp/blob/154d9248b54e3888797fd81f93adc4700a75509a/neural_sp/models/seq2seq/decoders/ctc.py#L625 
'''

LOG_0 = -1e10
LOG_1 = 0

def np2tensor(array, device=None):
    """Convert form np.ndarray to torch.Tensor.
    Args:
        array (np.ndarray): A tensor of any sizes
    Returns:
        tensor (torch.Tensor):
    """
    tensor = torch.from_numpy(array).to(device)
    return tensor
def pad_list(xs, pad_value=0., pad_left=False):
    """Convert list of Tensors to a single Tensor with padding.
    Args:
        xs (list): A list of length `[B]`, which contains Tensors of size `[T, input_size]`
        pad_value (float):
        pad_left (bool):
    Returns:
        xs_pad (FloatTensor): `[B, T, input_size]`
    """
    bs = len(xs)
    max_time = max(x.size(0) for x in xs)
    xs_pad = xs[0].new_zeros(bs, max_time, * xs[0].size()[1:]).fill_(pad_value)
    for b in range(bs):
        if len(xs[b]) == 0:
            continue
        if pad_left:
            xs_pad[b, -xs[b].size(0):] = xs[b]
        else:
            xs_pad[b, :xs[b].size(0)] = xs[b]
    return xs_pad
class CTCForcedAligner(object):
    def __init__(self, blank=0, char_list=None):
        self.blank = blank
        self.symbols = [0, 1, 2, 29, 30]
        self.char_list = char_list
    def __call__(self, logits, elens, ys):
        """Forced alignment with references.
        Args:
            logits (FloatTensor): `[B, T, vocab]`
            elens (List): length `[B]`
            ys (List): length `[B]`, each of which contains a list of size `[L]`
            ylens (List): length `[B]`
        Returns:
            trigger_points (IntTensor): `[B, L]`
        """
        ylens = np2tensor(np.fromiter([len(y) for y in ys], dtype=np.int32))
        with torch.no_grad():
            ys = [np2tensor(np.fromiter(y, dtype=np.int64), logits.device) for y in ys]
            ys_in_pad = pad_list(ys, 0)
            # zero padding
            mask = make_pad_mask(elens.to(logits.device))
            mask = mask.unsqueeze(2).expand_as(logits)
            logits = logits.masked_fill_(mask == 0, LOG_0)
            log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)  # `[T, B, vocab]`

            trigger_points = self.align(log_probs, elens, ys_in_pad, ylens)
        return trigger_points

    def align(self, log_probs, elens, ys, ylens, add_eos=True):
        """Calculte the best CTC alignment with the forward-backward algorithm.
        Args:
            log_probs (FloatTensor): `[T, B, vocab]`
            elens (FloatTensor): `[B]`
            ys (FloatTensor): `[B, L]`
            ylens (FloatTensor): `[B]`
            add_eos (bool): Use the last time index as a boundary corresponding to <eos>
        Returns:
            trigger_points (IntTensor): `[B, L]`
        """
        xmax, bs, vocab = log_probs.size()

        path = _label_to_path(ys, self.blank)
        path_lens = 2 * ylens.long() + 1

        ymax = ys.size(1)

        max_path_len = path.size(1)
        assert ys.size() == (bs, ymax), ys.size()
        assert path.size() == (bs, ymax * 2 + 1)

        alpha = log_probs.new_zeros(bs, max_path_len).fill_(LOG_0)
        alpha[:, 0] = LOG_1
        beta = alpha.clone()
        gamma = alpha.clone()

        batch_index = torch.arange(bs, dtype=torch.int64).unsqueeze(1)
        frame_index = torch.arange(xmax, dtype=torch.int64).unsqueeze(1).unsqueeze(2)
        log_probs_fwd_bwd = log_probs[frame_index, batch_index, path]
        same_transition = (path[:, :-2] == path[:, 2:])
        outside = torch.arange(max_path_len, dtype=torch.int64) >= path_lens.unsqueeze(1)
        log_probs_gold = log_probs[:, batch_index, path]

        # forward algorithm
        for t in range(xmax):
            alpha = _computes_transition(alpha, same_transition, outside,
                                         log_probs_fwd_bwd[t], log_probs_gold[t])

        # backward algorithm
        r_path = _flip_path(path, path_lens)
        log_probs_inv = _flip_label_probability(log_probs, elens.long())  # `[T, B, vocab]`
        log_probs_fwd_bwd = _flip_path_probability(log_probs_fwd_bwd, elens.long(), path_lens)  # `[T, B, 2*L+1]`
        r_same_transition = (r_path[:, :-2] == r_path[:, 2:])
        log_probs_inv_gold = log_probs_inv[:, batch_index, r_path]
        for t in range(xmax):
            beta = _computes_transition(beta, r_same_transition, outside,
                                        log_probs_fwd_bwd[t], log_probs_inv_gold[t])

        # pick up the best CTC path
        best_aligns = log_probs.new_zeros((bs, xmax), dtype=torch.int64)

        # forward algorithm
        log_probs_fwd_bwd = _flip_path_probability(log_probs_fwd_bwd, elens.long(), path_lens)
        for t in range(xmax):
            gamma = _computes_transition(gamma, same_transition, outside,
                                         log_probs_fwd_bwd[t], log_probs_gold[t],
                                         skip_accum=True)

            # select paths where gamma is valid
            log_probs_fwd_bwd[t] = log_probs_fwd_bwd[t].masked_fill_(gamma == LOG_0, LOG_0)

            # pick up the best alignment
            offsets = log_probs_fwd_bwd[t].argmax(1)
            for b in range(bs):
                if t <= elens[b] - 1:
                    token_idx = path[b, offsets[b]]
                    best_aligns[b, t] = token_idx

            # remove the rest of paths
            gamma = log_probs.new_zeros(bs, max_path_len).fill_(LOG_0)
            for b in range(bs):
                gamma[b, offsets[b]] = LOG_1

        # pick up trigger points
        trigger_aligns = torch.zeros((bs, xmax), dtype=torch.int64)
        trigger_aligns_avg = torch.zeros((bs, xmax), dtype=torch.int64)
        trigger_points = log_probs.new_zeros((bs, ymax + 1), dtype=torch.int32)  # +1 for <eos>
        for b in range(bs):
            n_triggers = 0
            if add_eos:
                trigger_points[b, ylens[b]] = elens[b] - 1
                # NOTE: use the last time index as a boundary corresponding to <eos>
                # Otherwise, index: 0 is used for <eos>
            last_token_idx = None
            count = 0
            for t in range(elens[b]):
                token_idx = best_aligns[b, t]
                if token_idx in self.symbols:
                    trigger_aligns_avg[b, t] = last_token_idx if last_token_idx else 0
                    count += 1
                if token_idx == self.blank:
                    continue
                if not (t == 0 or token_idx != best_aligns[b, t - 1]):
                    continue
                
                # NOTE: select the most left trigger points
                trigger_aligns[b, t] = token_idx
                last_token_idx = token_idx
                trigger_points[b, n_triggers] = t
                n_triggers += 1

        assert ylens.sum() == (trigger_aligns != 0).sum()
        
        return trigger_aligns_avg



def _flip_label_probability(log_probs, xlens):
    """Flips a label probability matrix.
    This function rotates a label probability matrix and flips it.
    ``log_probs[i, b, l]`` stores log probability of label ``l`` at ``i``-th
    input in ``b``-th batch.
    The rotated matrix ``r`` is defined as
    ``r[i, b, l] = log_probs[i + xlens[b], b, l]``
    Args:
        cum_log_prob (FloatTensor): `[T, B, vocab]`
        xlens (LongTensor): `[B]`
    Returns:
        FloatTensor: `[T, B, vocab]`
    """
    xmax, bs, vocab = log_probs.size()
    rotate = (torch.arange(xmax, dtype=torch.int64)[:, None] + xlens) % xmax
    return torch.flip(log_probs[rotate[:, :, None],
                                torch.arange(bs, dtype=torch.int64)[None, :, None],
                                torch.arange(vocab, dtype=torch.int64)[None, None, :]], dims=[0])


def _flip_path_probability(cum_log_prob, xlens, path_lens):
    """Flips a path probability matrix.
    This function returns a path probability matrix and flips it.
    ``cum_log_prob[i, b, t]`` stores log probability at ``i``-th input and
    at time ``t`` in a output sequence in ``b``-th batch.
    The rotated matrix ``r`` is defined as
    ``r[i, j, k] = cum_log_prob[i + xlens[j], j, k + path_lens[j]]``
    Args:
        cum_log_prob (FloatTensor): `[T, B, 2*L+1]`
        xlens (LongTensor): `[B]`
        path_lens (LongTensor): `[B]`
    Returns:
        FloatTensor: `[T, B, 2*L+1]`
    """
    xmax, bs, max_path_len = cum_log_prob.size()
    rotate_input = ((torch.arange(xmax, dtype=torch.int64)[:, None] + xlens) % xmax)
    rotate_label = ((torch.arange(max_path_len, dtype=torch.int64) + path_lens[:, None]) % max_path_len)
    return torch.flip(cum_log_prob[rotate_input[:, :, None],
                                   torch.arange(bs, dtype=torch.int64)[None, :, None],
                                   rotate_label], dims=[0, 2])


def _computes_transition(seq_log_prob, same_transition, outside,
                         cum_log_prob, log_prob_yt, skip_accum=False):
    bs, max_path_len = seq_log_prob.size()
    mat = seq_log_prob.new_zeros(3, bs, max_path_len).fill_(LOG_0)
    mat[0, :, :] = seq_log_prob
    mat[1, :, 1:] = seq_log_prob[:, :-1]
    mat[2, :, 2:] = seq_log_prob[:, :-2]
    # disable transition between the same symbols
    # (including blank-to-blank)
    mat[2, :, 2:][same_transition] = LOG_0
    seq_log_prob = torch.logsumexp(mat, dim=0)  # overwrite
    seq_log_prob[outside] = LOG_0
    if not skip_accum:
        cum_log_prob += seq_log_prob
    seq_log_prob += log_prob_yt
    return seq_log_prob

def make_pad_mask(seq_lens):
    """Make mask for padding.
    Args:
        seq_lens (IntTensor): `[B]`
    Returns:
        mask (IntTensor): `[B, T]`
    """
    bs = seq_lens.size(0)
    max_time = seq_lens.max()
    seq_range = torch.arange(0, max_time, dtype=torch.int32, device=seq_lens.device)
    seq_range = seq_range.unsqueeze(0).expand(bs, max_time)
    mask = seq_range < seq_lens.unsqueeze(-1)
    return mask

def _label_to_path(labels, blank):
    path = labels.new_zeros(labels.size(0), labels.size(1) * 2 + 1).fill_(blank).long()
    path[:, 1::2] = labels
    return path
def _flip_path(path, path_lens):
    """Flips label sequence.
    This function rotates a label sequence and flips it.
    ``path[b, t]`` stores a label at time ``t`` in ``b``-th batch.
    The rotated matrix ``r`` is defined as
    ``r[b, t] = path[b, t + path_lens[b]]``
    .. ::
       a b c d .     . a b c d    d c b a .
       e f . . .  -> . . . e f -> f e . . .
       g h i j k     g h i j k    k j i h g
    Args:
        path (FloatTensor): `[B, 2*L+1]`
        path_lens (LongTensor): `[B]`
    Returns:
        FloatTensor: `[B, 2*L+1]`
    """
    bs = path.size(0)
    max_path_len = path.size(1)
    rotate = (torch.arange(max_path_len) + path_lens[:, None]) % max_path_len
    return torch.flip(path[torch.arange(bs, dtype=torch.int64)[:, None], rotate], dims=[1])
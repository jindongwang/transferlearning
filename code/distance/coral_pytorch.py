# Compute CORAL loss using pytorch
# Reference: DCORAL: Correlation Alignment for Deep Domain Adaptation, ECCV-16.

import torch

def CORAL_loss(source, target):
    d = source.data.shape[1]
    ns, nt = source.data.shape[0], target.data.shape[0]
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm / (ns - 1)

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt / (nt - 1)

    # frobenius norm between source and target
    loss = torch.mul((xc - xct), (xc - xct))
    loss = torch.sum(loss) / (4*d*d)
    return loss

# Another implementation:
# Two implementations are the same. Just different formulation format.
# def CORAL(source, target):
#     d = source.size(1)
#     ns, nt = source.size(0), target.size(0)
#     # source covariance
#     tmp_s = torch.ones((1, ns)).to(DEVICE) @ source
#     cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

#     # target covariance
#     tmp_t = torch.ones((1, nt)).to(DEVICE) @ target
#     ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

#     # frobenius norm
#     loss = torch.norm(cs - ct, p='fro').pow(2)
#     loss = loss / (4 * d * d)
#     return loss
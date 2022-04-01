import torch.nn as nn

def cosine(source, target):
    source, target = source.mean(0), target.mean(0)
    cos = nn.CosineSimilarity(dim=0)
    loss = cos(source, target)
    return loss.mean()
# Deep Coral

A PyTorch implementation of '[Deep CORAL Correlation Alignment for Deep Domain Adaptation](https://arxiv.org/pdf/1607.01719.pdf)'.
The contributions of this paper are summarized as fol-
lows.
* They extend CORAL to incorporate it directly into deep networks by constructing a differentiable loss function that minimizes the difference between source and target correlations–the CORAL loss.
* Compared to CORAL, Deep CORAL approach learns a non-linear transformation that is more powerful and also works seamlessly with deep CNNs.
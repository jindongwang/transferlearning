import torch
import torch.nn as nn


class OZELoss(nn.Module):
    """Custom loss for TRNSys metamodel.

    Compute, for temperature and consumptions, the intergral of the squared differences
    over time. Sum the log with a coeficient ``alpha``.

    .. math::
        \Delta_T = \sqrt{\int (y_{est}^T - y^T)^2}

        \Delta_Q = \sqrt{\int (y_{est}^Q - y^Q)^2}

        loss = log(1 + \Delta_T) + \\alpha \cdot log(1 + \Delta_Q)

    Parameters:
    -----------
    alpha:
        Coefficient for consumption. Default is ``0.3``.
    """

    def __init__(self, reduction: str = 'mean', alpha: float = 0.3):
        super().__init__()

        self.alpha = alpha
        self.reduction = reduction

        self.base_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_true: torch.Tensor,
                y_pred: torch.Tensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Parameters
        ----------
        y_true:
            Target value.
        y_pred:
            Estimated value.

        Returns
        -------
        Loss as a tensor with gradient attached.
        """
        delta_Q = self.base_loss(y_pred[..., :-1], y_true[..., :-1])
        delta_T = self.base_loss(y_pred[..., -1], y_true[..., -1])

        if self.reduction == 'none':
            delta_Q = delta_Q.mean(dim=(1, 2))
            delta_T = delta_T.mean(dim=(1))

        return torch.log(1 + delta_T) + self.alpha * torch.log(1 + delta_Q)

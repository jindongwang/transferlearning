import torch
import torch.nn as nn
import torch.nn.functional as F

class Mine_estimator(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512):
        super(Mine_estimator, self).__init__()
        self.mine_model = Mine(input_dim, hidden_dim)

    def forward(self, X, Y):
        Y_shffle = Y[torch.randperm(len(Y))]
        loss_joint = self.mine_model(X, Y)
        loss_marginal = self.mine_model(X, Y_shffle)
        ret = torch.mean(loss_joint) - \
            torch.log(torch.mean(torch.exp(loss_marginal)))
        loss = -ret
        return loss


class Mine(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512):
        super(Mine, self).__init__()
        self.fc1_x = nn.Linear(input_dim, hidden_dim)
        self.fc1_y = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, y):
        h1 = F.leaky_relu(self.fc1_x(x)+self.fc1_y(y))
        h2 = self.fc2(h1)
        return h2
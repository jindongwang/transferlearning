import torch
import torch.nn as nn
from base.loss_transfer import TransferLoss
import torch.nn.functional as F


class AdaRNN(nn.Module):
    """
    model_type:  'Boosting', 'AdaRNN'
    """

    def __init__(self, use_bottleneck=False, bottleneck_width=256, n_input=128, n_hiddens=[64, 64], n_output=6, dropout=0.0, len_seq=9, model_type='AdaRNN', trans_loss='mmd'):
        super(AdaRNN, self).__init__()
        self.use_bottleneck = use_bottleneck
        self.n_input = n_input
        self.num_layers = len(n_hiddens)
        self.hiddens = n_hiddens
        self.n_output = n_output
        self.model_type = model_type
        self.trans_loss = trans_loss
        self.len_seq = len_seq
        in_size = self.n_input

        features = nn.ModuleList()
        for hidden in n_hiddens:
            rnn = nn.GRU(
                input_size=in_size,
                num_layers=1,
                hidden_size=hidden,
                batch_first=True,
                dropout=dropout
            )
            features.append(rnn)
            in_size = hidden
        self.features = nn.Sequential(*features)

        if use_bottleneck == True:  # finance
            self.bottleneck = nn.Sequential(
                nn.Linear(n_hiddens[-1], bottleneck_width),
                nn.Linear(bottleneck_width, bottleneck_width),
                nn.BatchNorm1d(bottleneck_width),
                nn.ReLU(),
                nn.Dropout(),
            )
            self.bottleneck[0].weight.data.normal_(0, 0.005)
            self.bottleneck[0].bias.data.fill_(0.1)
            self.bottleneck[1].weight.data.normal_(0, 0.005)
            self.bottleneck[1].bias.data.fill_(0.1)
            self.fc = nn.Linear(bottleneck_width, n_output)
            torch.nn.init.xavier_normal_(self.fc.weight)
        else:
            self.fc_out = nn.Linear(n_hiddens[-1], self.n_output)

        if self.model_type == 'AdaRNN':
            gate = nn.ModuleList()
            for i in range(len(n_hiddens)):
                gate_weight = nn.Linear(
                    len_seq * self.hiddens[i]*2, len_seq)
                gate.append(gate_weight)
            self.gate = gate

            bnlst = nn.ModuleList()
            for i in range(len(n_hiddens)):
                bnlst.append(nn.BatchNorm1d(len_seq))
            self.bn_lst = bnlst
            self.softmax = torch.nn.Softmax(dim=0)
            self.init_layers()

    def init_layers(self):
        for i in range(len(self.hiddens)):
            self.gate[i].weight.data.normal_(0, 0.05)
            self.gate[i].bias.data.fill_(0.0)

    def forward_pre_train(self, x, len_win=0):
        out = self.gru_features(x)
        fea = out[0]
        if self.use_bottleneck == True:
            fea_bottleneck = self.bottleneck(fea[:, -1, :])
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            fc_out = self.fc_out(fea[:, -1, :]).squeeze()

        out_list_all, out_weight_list = out[1], out[2]
        out_list_s, out_list_t = self.get_features(out_list_all)
        loss_transfer = torch.zeros((1,)).cuda()
        for i in range(len(out_list_s)):
            criterion_transder = TransferLoss(
                loss_type=self.trans_loss, input_dim=out_list_s[i].shape[2])
            h_start = 0 
            for j in range(h_start, self.len_seq, 1):
                i_start = j - len_win if j - len_win >= 0 else 0
                i_end = j + len_win if j + len_win < self.len_seq else self.len_seq - 1
                for k in range(i_start, i_end + 1):
                    weight = out_weight_list[i][j] if self.model_type == 'AdaRNN' else 1 / (
                        self.len_seq - h_start) * (2 * len_win + 1)
                    loss_transfer = loss_transfer + weight * criterion_transder.compute(
                        out_list_s[i][:, j, :], out_list_t[i][:, k, :])
        return fc_out, loss_transfer, out_weight_list

    def gru_features(self, x, predict=False):
        x_input = x
        out = None
        out_lis = []
        out_weight_list = [] if (
             self.model_type == 'AdaRNN') else None
        for i in range(self.num_layers):
            out, _ = self.features[i](x_input.float())
            x_input = out
            out_lis.append(out)
            if self.model_type == 'AdaRNN' and predict == False:
                out_gate = self.process_gate_weight(x_input, i)
                out_weight_list.append(out_gate)
        return out, out_lis, out_weight_list

    def process_gate_weight(self, out, index):
        x_s = out[0: int(out.shape[0]//2)]
        x_t = out[out.shape[0]//2: out.shape[0]]
        x_all = torch.cat((x_s, x_t), 2)
        x_all = x_all.view(x_all.shape[0], -1)
        weight = torch.sigmoid(self.bn_lst[index](
            self.gate[index](x_all.float())))
        weight = torch.mean(weight, dim=0)
        res = self.softmax(weight).squeeze()
        return res

    def get_features(self, output_list):
        fea_list_src, fea_list_tar = [], []
        for fea in output_list:
            fea_list_src.append(fea[0: fea.size(0) // 2])
            fea_list_tar.append(fea[fea.size(0) // 2:])
        return fea_list_src, fea_list_tar

    # For Boosting-based
    def forward_Boosting(self, x, weight_mat=None):
        out = self.gru_features(x)
        fea = out[0]
        if self.use_bottleneck:
            fea_bottleneck = self.bottleneck(fea[:, -1, :])
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            fc_out = self.fc_out(fea[:, -1, :]).squeeze()

        out_list_all = out[1]
        out_list_s, out_list_t = self.get_features(out_list_all)
        loss_transfer = torch.zeros((1,)).cuda()
        if weight_mat is None:
            weight = (1.0 / self.len_seq *
                      torch.ones(self.num_layers, self.len_seq)).cuda()
        else:
            weight = weight_mat
        dist_mat = torch.zeros(self.num_layers, self.len_seq).cuda()
        for i in range(len(out_list_s)):
            criterion_transder = TransferLoss(
                loss_type=self.trans_loss, input_dim=out_list_s[i].shape[2])
            for j in range(self.len_seq):
                loss_trans = criterion_transder.compute(
                    out_list_s[i][:, j, :], out_list_t[i][:, j, :])
                loss_transfer = loss_transfer + weight[i, j] * loss_trans
                dist_mat[i, j] = loss_trans
        return fc_out, loss_transfer, dist_mat, weight

    # For Boosting-based
    def update_weight_Boosting(self, weight_mat, dist_old, dist_new):
        epsilon = 1e-12
        dist_old = dist_old.detach()
        dist_new = dist_new.detach()
        ind = dist_new > dist_old + epsilon
        weight_mat[ind] = weight_mat[ind] * \
            (1 + torch.sigmoid(dist_new[ind] - dist_old[ind]))
        weight_norm = torch.norm(weight_mat, dim=1, p=1)
        weight_mat = weight_mat / weight_norm.t().unsqueeze(1).repeat(1, self.len_seq)
        return weight_mat

    def predict(self, x):
        out = self.gru_features(x, predict=True)
        fea = out[0]
        if self.use_bottleneck == True:
            fea_bottleneck = self.bottleneck(fea[:, -1, :])
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            fc_out = self.fc_out(fea[:, -1, :]).squeeze()
        return fc_out

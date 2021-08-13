from base.loss import adv_loss, CORAL, kl_js, mmd, mutual_info, cosine, pairwise_dist


class TransferLoss(object):
    def __init__(self, loss_type='cosine', input_dim=512):
        """
        Supported loss_type: mmd(mmd_lin), mmd_rbf, coral, cosine, kl, js, mine, adv
        """
        self.loss_type = loss_type
        self.input_dim = input_dim

    def compute(self, X, Y):
        """Compute adaptation loss

        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix

        Returns:
            [tensor] -- transfer loss
        """
        if self.loss_type == 'mmd_lin' or self.loss_type =='mmd':
            mmdloss = mmd.MMD_loss(kernel_type='linear')
            loss = mmdloss(X, Y)
        elif self.loss_type == 'coral':
            loss = CORAL(X, Y)
        elif self.loss_type == 'cosine' or self.loss_type == 'cos':
            loss = 1 - cosine(X, Y)
        elif self.loss_type == 'kl':
            loss = kl_js.kl_div(X, Y)
        elif self.loss_type == 'js':
            loss = kl_js.js(X, Y)
        elif self.loss_type == 'mine':
            mine_model = mutual_info.Mine_estimator(
                input_dim=self.input_dim, hidden_dim=60).cuda()
            loss = mine_model(X, Y)
        elif self.loss_type == 'adv':
            loss = adv_loss.adv(X, Y, input_dim=self.input_dim, hidden_dim=32)
        elif self.loss_type == 'mmd_rbf':
            mmdloss = mmd.MMD_loss(kernel_type='rbf')
            loss = mmdloss(X, Y)
        elif self.loss_type == 'pairwise':
            pair_mat = pairwise_dist(X, Y)
            import torch
            loss = torch.norm(pair_mat)

        return loss

if __name__ == "__main__":
    import torch
    trans_loss = TransferLoss('adv')
    a = (torch.randn(5,512) * 10).cuda()
    b = (torch.randn(5,512) * 10).cuda()
    print(trans_loss.compute(a, b))

from loss_funcs.adv import *

class DAANLoss(AdversarialLoss, LambdaSheduler):
    def __init__(self, num_class, gamma=1.0, max_iter=1000, **kwargs):
        super(DAANLoss, self).__init__(gamma=gamma, max_iter=max_iter, **kwargs)
        self.num_class = num_class
        self.local_classifiers = torch.nn.ModuleList()
        for _ in range(num_class):
            self.local_classifiers.append(Discriminator())

        self.d_g, self.d_l = 0, 0
        self.dynamic_factor = 0.5

    def forward(self, source, target, source_logits, target_logits):
        lamb = self.lamb()
        self.step()
        source_loss_g = self.get_adversarial_result(source, True, lamb)
        target_loss_g = self.get_adversarial_result(target, False, lamb)
        source_loss_l = self.get_local_adversarial_result(source, source_logits, True, lamb)
        target_loss_l = self.get_local_adversarial_result(target, target_logits, False, lamb)
        global_loss = 0.5 * (source_loss_g + target_loss_g) * 0.05
        local_loss = 0.5 * (source_loss_l + target_loss_l) * 0.01

        self.d_g = self.d_g + 2 * (1 - 2 * global_loss.cpu().item())
        self.d_l = self.d_l + 2 * (1 - 2 * (local_loss / self.num_class).cpu().item())

        adv_loss = (1 - self.dynamic_factor) * global_loss + self.dynamic_factor * local_loss
        return adv_loss
    
    def get_local_adversarial_result(self, x, logits, c, source=True, lamb=1.0):
        loss_fn = nn.BCELoss()
        x = ReverseLayerF.apply(x, lamb)
        loss_adv = 0.0

        for c in range(self.num_class):
            logits_c = logits[:, c].reshape((logits.shape[0],1)) # (B, 1)
            features_c = logits_c * x
            domain_pred = self.local_classifiers[c](features_c)
            device = domain_pred.device
            if source:
                domain_label = torch.ones(len(x), 1).long()
            else:
                domain_label = torch.zeros(len(x), 1).long()
            loss_adv = loss_adv + loss_fn(domain_pred, domain_label.float().to(device))
        return loss_adv
    
    def update_dynamic_factor(self, epoch_length):
        if self.d_g == 0 and self.d_l == 0:
            self.dynamic_factor = 0.5
        else:
            self.d_g = self.d_g / epoch_length
            self.d_l = self.d_l / epoch_length
            self.dynamic_factor = 1 - self.d_g / (self.d_g + self.d_l)
        self.d_g, self.d_l = 0, 0

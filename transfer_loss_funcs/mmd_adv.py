import torch.nn as nn
from transfer_loss_funcs.mmd import MMDLoss
from transfer_loss_funcs.adv import AdversarialLoss


# ---------------------------------adv+mmd---------------------------------- #
class MMD_ADVLoss(nn.Module):
    def __init__(self, **kwargs):
        super(MMD_ADVLoss, self).__init__()
        self.mmdloss = MMDLoss(**kwargs)
        self.advloss = AdversarialLoss(**kwargs)

    def forward(self, source, target, **kwargs):
        mmd = self.mmdloss(source, target, **kwargs)
        adv = self.advloss(source, target, **kwargs)
        return mmd + adv


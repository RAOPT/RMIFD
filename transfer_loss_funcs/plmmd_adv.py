import torch.nn as nn
import copy
from transfer_loss_funcs.lmmd import LMMDLoss
from transfer_loss_funcs.plmmd import PLMMDLoss
from transfer_loss_funcs.adv import AdversarialLoss


# ---------------------------------adv+mmd---------------------------------- #
class PLMMD_ADVLoss(nn.Module):
    def __init__(self, **kwargs):
        super(PLMMD_ADVLoss, self).__init__()
        self.plmmdloss = PLMMDLoss(**kwargs)
        self.advloss = AdversarialLoss(**kwargs)

    def forward(self, source, target, **kwargs):
        plmmd = self.plmmdloss(source, target, **kwargs)
        adv = self.advloss(source, target)
        return plmmd + adv


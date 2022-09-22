import torch.nn as nn
import copy
from transfer_loss_funcs.lmmd import LMMDLoss
from transfer_loss_funcs.plmmd import PLMMDLoss
from transfer_loss_funcs.adv import AdversarialLoss


# ---------------------------------adv+mmd---------------------------------- #
class LMMD_ADVLoss(nn.Module):
    def __init__(self, **kwargs):
        super(LMMD_ADVLoss, self).__init__()
        self.lmmdloss = LMMDLoss(**kwargs)
        self.advloss = AdversarialLoss(**kwargs)

    def forward(self, source, target, **kwargs):
        lmmd = self.lmmdloss(source, target, **kwargs)
        adv = self.advloss(source, target)
        return lmmd + adv


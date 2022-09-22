import torch, networks_basic
import torch.nn as nn
from x_tools.transfer_loss import TransferLoss

transfer_loss_types = ["mmd", "adv", "lmmd", "coral", "bnm", "mmd_adv", "lmmd_adv", "plmmd_adv", "daan"]

################################# ########################## ############################################
################################# ##### transfernet1d ###### ############################################
################################# ########################## ############################################
class transfernet1d(nn.Module):
    """
    :param:
    num_class:int
    basic_net:str
    transfer_loss:str
    :return:
    cls_loss
    transfer_loss
    """
    def __init__(self, num_class, basic_net="cnn1d", transfer_loss="mmd", **kwargs):
        super(transfernet1d, self).__init__()
        self.num_class = num_class
        self.basic_net = getattr(networks_basic, basic_net)(out_channel=num_class, is_tl=True)
        # self.classifier = nn.Linear(self.basic_net.out_dim(), num_class)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.basic_net.out_dim(), num_class)
        )
        self.transfer_loss = transfer_loss
        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": 1000,
            "num_class": num_class,
            "out_dim": self.classifier[-1].in_features     # for adv: Discriminator in_features;
        }
        self.adapt_loss_func = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, source, target, source_label):
        source = self.basic_net(source)
        target = self.basic_net(target)
        source_cls = self.classifier(source)

        # classification loss:
        cls_loss = self.criterion(source_cls, source_label)

        # transfer loss
        kwargs = {}
        # kwargs["input_dim"] = self.classifier[-1].in_features
        if self.transfer_loss == "lmmd":
            kwargs['source_label'] = source_label
            target_clf = self.classifier(target)
            kwargs['target_logits'] = nn.Softmax(1)(target_clf)
        elif self.transfer_loss == "daan":
            source_clf = self.classifier(source)
            kwargs['source_logits'] = nn.Softmax(1)(source_clf)
            target_clf = self.classifier(target)
            kwargs['target_logits'] = nn.Softmax(1)(target_clf)
        elif self.transfer_loss == "lmmd_adv":
            kwargs['source_label'] = source_label
            target_clf = self.classifier(target)
            kwargs['target_logits'] = nn.Softmax(1)(target_clf)
        elif self.transfer_loss == "plmmd_adv":
            kwargs['source_label'] = source_label
            target_clf = self.classifier(target)
            kwargs['target_logits'] = nn.Softmax(1)(target_clf)
        elif self.transfer_loss == 'bnm':
            tar_clf = self.classifier(target)
            target = nn.Softmax(dim=1)(tar_clf)
        elif self.transfer_loss not in transfer_loss_types:
            # print("WARNING: No valid transfer loss function is used.")
            return cls_loss, torch.tensor(0)                                    # for source only, return 0 as transfer_loss
        transfer_loss = self.adapt_loss_func(source, target, **kwargs)
        return cls_loss, transfer_loss

    def get_params(self, initial_lr):
        params = [
            {"params": self.basic_net.parameters(), "lr": 0.1 * initial_lr},
            {"params": self.classifier.parameters(), "lr": initial_lr}
        ]
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss_func.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "mmd_adv":
            params.append(
                {'params': self.adapt_loss_func.loss_func.advloss.domain_classifier.parameters(),
                 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "lmmd_adv":
            params.append(
                {"params": self.adapt_loss_func.loss_func.advloss.domain_classifier.parameters(),
                 "lr": 1.0 * initial_lr}
            )
        elif self.transfer_loss == "plmmd_adv":
            params.append(
                {"params": self.adapt_loss_func.loss_func.plmmdloss.parameters(), "lr": 1.0 * initial_lr}
            )
            params.append(
                {"params": self.adapt_loss_func.loss_func.advloss.domain_classifier.parameters(),
                 "lr": 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss_func.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss_func.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        features = self.basic_net(x)
        cls = self.classifier(features)
        return cls

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass


################################# ########################## ############################################
################################# #### transfernet_datn #### ############################################
################################# ########################## ############################################
class transfernet_datn(nn.Module):
    """
    :param:
    num_class:int
    basic_net:str
    transfer_loss:str
    :return:
    cls_loss
    transfer_loss
    """
    def __init__(self, num_class, basic_net="cnn1d", transfer_loss="mmd", **kwargs):
        super(transfernet_datn, self).__init__()
        self.num_class = num_class
        self.basic_net_s = getattr(networks_basic, basic_net)(out_channel=num_class, is_tl=True)
        self.basic_net_t = getattr(networks_basic, basic_net)(out_channel=num_class, is_tl=True)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.basic_net_s.out_dim(), num_class)
        )
        self.source_net = nn.Sequential(
            self.basic_net_s,
            self.classifier
        )
        self.transfer_loss = transfer_loss
        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": 1000,
            "num_class": num_class,
            "out_dim": self.basic_net_s.out_dim()     # for adv: Discriminator in_features;
        }
        self.adapt_loss_func = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_source_net_weight(self):
        # source_net_params = [{"params": self.source_net.parameters(), "lr": initial_lr}]
        source_net_params = self.source_net.parameters()
        return source_net_params

    def load_source_net_weight(self, path):
        self.source_net.load_state_dict(torch.load(path))

    def forward(self, source, target, source_label):
        source = self.basic_net_s(source)
        target = self.basic_net_t(target)
        source_cls = self.classifier(source)

        # classification loss:  # in datn this could be ignored
        cls_loss = self.criterion(source_cls, source_label)

        # transfer loss
        kwargs = {}
        # kwargs["input_dim"] = self.classifier[-1].in_features
        if self.transfer_loss == "lmmd":
            kwargs['source_label'] = source_label
            target_clf = self.classifier(target)
            kwargs['target_logits'] = nn.Softmax(1)(target_clf)
        elif self.transfer_loss == "daan":
            source_clf = self.classifier(source)
            kwargs['source_logits'] = nn.Softmax(1)(source_clf)
            target_clf = self.classifier(target)
            kwargs['target_logits'] = nn.Softmax(1)(target_clf)
        elif self.transfer_loss == "lmmd_adv":
            kwargs['source_label'] = source_label
            target_clf = self.classifier(target)
            kwargs['target_logits'] = nn.Softmax(1)(target_clf)
        elif self.transfer_loss == "plmmd_adv":
            kwargs['source_label'] = source_label
            target_clf = self.classifier(target)
            kwargs['target_logits'] = nn.Softmax(1)(target_clf)
        elif self.transfer_loss == 'bnm':
            tar_clf = self.classifier(target)
            target = nn.Softmax(dim=1)(tar_clf)
        else:
            # print("WARNING: No valid transfer loss function is used.")
            return cls_loss, torch.tensor(0)                                    # for source only, return 0 as transfer_loss
        transfer_loss = self.adapt_loss_func(source, target, **kwargs)
        return cls_loss, transfer_loss

    def get_params(self, initial_lr):
        params = [
            # {"params": self.basic_net_s.parameters(), "lr": 0.1 * initial_lr},
            {"params": self.basic_net_t.parameters(), "lr": 0.1 * initial_lr},
            # {"params": self.classifier.parameters(), "lr": initial_lr}
        ]
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss_func.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss_func.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss_func.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        features = self.basic_net_t(x)
        cls = self.classifier(features)
        return cls

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass

################################# ########################## ############################################
################################# transfernet1d_multi_branch ############################################
################################# ########################## ############################################
class transfernet1d_multi_branch(nn.Module):
    """
    :param:
    num_class:int
    basic_net:str
    transfer_loss:str
    :return:
    cls_loss
    transfer_loss
    """
    def __init__(self, num_class, basic_net="cnn1d", transfer_loss="mmd", **kwargs):
        super(transfernet1d_multi_branch, self).__init__()
        self.num_class = num_class
        self.basic_net_s = getattr(networks_basic, basic_net)(out_channel=num_class, is_tl=True)    # for time 
        self.basic_net_t = getattr(networks_basic, basic_net)(out_channel=num_class, is_tl=True)    # for fft
        # branch name: s t u v w...
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.basic_net_s.out_dim() + self.basic_net_t.out_dim(), num_class)
        )
        self.transfer_loss = transfer_loss
        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": 1000,
            "num_class": num_class,
            "out_dim": self.classifier[-1].in_features     # for adv: Discriminator in_features;
        }
        self.adapt_loss_func = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, source, target, source_label):
        data1_length = int(source.shape[-1] * 2 / 3)  # the length of fft is 1/2 of id;
        source_1d = self.basic_net_s(source[:, :, :data1_length])
        source_fft = self.basic_net_t(source[:, :, data1_length:])
        target_1d = self.basic_net_s(target[:, :, :data1_length])
        target_fft = self.basic_net_t(target[:, :, data1_length:])
        source = torch.cat([source_1d, source_fft], -1)
        target = torch.cat([target_1d, target_fft], -1)
        source_cls = self.classifier(source)

        # classification loss:  # in datn this could be ignored
        cls_loss = self.criterion(source_cls, source_label)

        # transfer loss
        kwargs = {}
        # kwargs["input_dim"] = self.classifier[-1].in_features
        if self.transfer_loss == "lmmd":
            kwargs['source_label'] = source_label
            target_clf = self.classifier(target)
            kwargs['target_logits'] = nn.Softmax(1)(target_clf)
        elif self.transfer_loss == "daan":
            source_clf = self.classifier(source)
            kwargs['source_logits'] = nn.Softmax(1)(source_clf)
            target_clf = self.classifier(target)
            kwargs['target_logits'] = nn.Softmax(1)(target_clf)
        elif self.transfer_loss == "lmmd_adv":
            kwargs['source_label'] = source_label
            target_clf = self.classifier(target)
            kwargs['target_logits'] = nn.Softmax(1)(target_clf)
        elif self.transfer_loss == "plmmd_adv":
            kwargs['source_label'] = source_label
            target_clf = self.classifier(target)
            kwargs['target_logits'] = nn.Softmax(1)(target_clf)
        elif self.transfer_loss == 'bnm':
            tar_clf = self.classifier(target)
            target = nn.Softmax(dim=1)(tar_clf)
        elif self.transfer_loss not in transfer_loss_types:
            # print("WARNING: No valid transfer loss function is used.")
            return cls_loss, torch.tensor(0)                                    # for source only, return 0 as transfer_loss
        transfer_loss = self.adapt_loss_func(source, target, **kwargs)
        return cls_loss, transfer_loss

    def get_params(self, initial_lr):
        params = [
            {"params": self.basic_net_s.parameters(), "lr": 0.1 * initial_lr},
            {"params": self.basic_net_t.parameters(), "lr": 0.1 * initial_lr},
            {"params": self.classifier.parameters(), "lr": initial_lr}
        ]
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss_func.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "mmd_adv":
            params.append(
                {'params': self.adapt_loss_func.loss_func.advloss.domain_classifier.parameters(),
                 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "lmmd_adv":
            params.append(
                {"params": self.adapt_loss_func.loss_func.advloss.domain_classifier.parameters(),
                 "lr": 1.0 * initial_lr}
            )
        elif self.transfer_loss == "plmmd_adv":
            params.append(
                {"params": self.adapt_loss_func.loss_func.plmmdloss.parameters(), "lr": 1.0 * initial_lr}
            )
            params.append(
                {"params": self.adapt_loss_func.loss_func.advloss.domain_classifier.parameters(),
                 "lr": 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss_func.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss_func.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        x_1d = self.basic_net_s(x[:, :, :int(x.shape[-1] * 2 / 3)])
        x_fft = self.basic_net_t(x[:, :, int(x.shape[-1] * 2 / 3):])
        features = torch.cat([x_1d, x_fft], -1)
        cls = self.classifier(features)
        return cls

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass

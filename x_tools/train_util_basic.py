import logging, time, warnings, math, torch, CNN_Datasets, x_tools, os
from torch import nn
from torch import optim
import networks_basic


class train_util_basic:
    def __init__(self, args):
        self.start_epoch = 0
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        [self.model, self.optimizer, self.scheduler, self.loader_dict] = [None for i in range(4)]

    def get_model(self):
        model = getattr(networks_basic, self.args.basic_net)(out_channel=self.args.num_class, is_tl=False)
        return model

    def get_optimizer(self, model):
        initial_lr = self.args.lr if not self.args.lr_scheduler_type else 1.0
        params = model.parameters()
        return optim.SGD(params, lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay,
                         nesterov=False)
        # return optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)

    def get_scheduler(self, optimizer, lr_scheduler_type="lambda"):
        if lr_scheduler_type == 'step':
            steps = [int(step) for step in self.args.steps.split(',')]
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=self.args.gamma)
        elif lr_scheduler_type == 'exp':
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, self.args.gamma)
        elif lr_scheduler_type == 'stepLR':
            steps = self.args.steps
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, steps, self.args.gamma)
            print("lr update size:", steps)
        elif lr_scheduler_type == 'fix':
            lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")
        return lr_scheduler

    def get_dataset(self):
        torch_dataset = getattr(CNN_Datasets, self.args.dataset)(self.args.data_dir, self.args.normalize_type)
        src_train_set, src_val_set = torch_dataset.data_prepare(self.args.src_wc)
        return {
            "st": src_train_set, "sv": src_val_set
        }

    def get_data_loader(self, ds: dict):
        torch_loader = torch.utils.data.DataLoader
        bs = self.args.batch_size
        return {
            "st": torch_loader(ds["st"], bs, shuffle=True),
            "sv": torch_loader(ds["sv"], bs, shuffle=False)
        }

    def set_up(self):
        self.model = self.get_model().to(self.device)
        self.optimizer = self.get_optimizer(self.model)
        self.scheduler = self.get_scheduler(self.optimizer, self.args.lr_scheduler_type)
        self.loader_dict = self.get_data_loader(self.get_dataset())

    def validate(self):
        self.model.eval()
        val_loss = x_tools.para_utils.AverageMeter()
        correct = 0
        criterion = nn.CrossEntropyLoss()
        target_loader = self.loader_dict["sv"]
        len_of_dataset = len(target_loader.dataset)
        with torch.no_grad():
            for data, label in target_loader:
                data, label = data.to(self.device), label.to(self.device)
                predict = self.model(data)
                val_loss.update(criterion(predict, label).item())
                out_fault_type = torch.max(predict, 1)[1]
                correct += torch.sum(out_fault_type == label)
        acc = 100. * correct/len_of_dataset
        return acc, val_loss.avg

    def train(self):
        save_dir = "checkpoint/" + "{}_{}_basic".format(
            self.args.dataset, self.args.basic_net
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.args.is_continue_train:
            self.load_checkpoint(save_dir + "/checkpoint.tar")

        len_of_loader = len(self.loader_dict["st"])
        best_acc, best_epoch = 0., 0
        for e in range(self.start_epoch, self.args.n_epoch):
            self.model.train()
            train_loss_cls = x_tools.para_utils.AverageMeter()
            if len_of_loader:
                iter_st = iter(self.loader_dict["st"])
            for n in range(len_of_loader):
                [st_data, st_label] = [i.to(self.device) for i in next(iter_st)]
                predict = self.model(st_data)
                cls_loss = self.criterion(predict, st_label)
                self.optimizer.zero_grad()
                cls_loss.backward()
                self.optimizer.step()
                train_loss_cls.update(cls_loss.item())
            info = "Epoch: [{:2d}/{}], cls_loss: {:.4f}".format(
                e, self.args.n_epoch, train_loss_cls.avg
            )
            val_acc, val_loss = self.validate()
            info += ", val_loss {:.4f}, val_acc {:.4f}, The current lr {:.6f}".format(
                val_loss, val_acc, self.optimizer.state_dict()["param_groups"][0]["lr"]
            )
            print(info)
            if self.args.is_save_checkpoint:
                self.save_checkpoint(e, save_dir)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = e
                if self.args.is_save_checkpoint:
                    self.save_checkpoint(e, save_dir, filename="checkpoint_max.tar")
            if self.scheduler:
                self.scheduler.step()
        print("Best acc:", best_acc)

    def save_checkpoint(self, epoch, save_dir, filename="checkpoint.tar"):
        """
        :param epoch: save epoch
        :param save_dir: decided by: {dataset, src_wc, tgt_wc, basic_net, transfer_loss}(5)
        :param filename: checkpoint.tar(latest), checkpoint_max.dir(best)
        :return: None
        """
        save_dict = {
            "epoch": epoch + 1,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "model_state_dict": self.model.state_dict()
        }
        torch.save(save_dict, os.path.join(save_dir, filename))

    def load_checkpoint(self, save_filename):
        """
        :param save_filename: should be the whole path including filename
        :return: None
        """
        if os.path.isfile(save_filename):
            checkpoint = torch.load(save_filename)
            self.start_epoch = checkpoint["epoch"]
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print("There has no checkpoint yet")

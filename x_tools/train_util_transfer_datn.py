import logging, math, torch, CNN_Datasets, x_tools, os
from torch import nn
from tqdm import tqdm
from torch import optim

import networks_transfer


class train_util_transfer_datn:
    def __init__(self, args):
        self.start_epoch = 0
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        [self.model, self.optimizer, self.scheduler, self.loader_dict] = [None for i in range(4)]

    def get_model(self):
        return networks_transfer.transfernet_datn(num_class=self.args.num_class, basic_net=self.args.basic_net,
                                                  transfer_loss=self.args.transfer_loss)

    def get_optimizer(self, model):
        initial_lr = self.args.lr if not self.args.lr_scheduler_type else 1.0
        params = model.get_params(self.args.lr)
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
            steps = int(self.args.steps)
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, steps, self.args.gamma)
        elif lr_scheduler_type == 'fix':
            lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")
        return lr_scheduler

    def get_dataset(self):
        torch_dataset = getattr(CNN_Datasets, self.args.dataset)(self.args.data_dir, self.args.normalize_type)
        src_train_set, src_val_set = torch_dataset.data_prepare(self.args.src_wc)
        tgt_train_set, tgt_val_set = torch_dataset.data_prepare(self.args.tgt_wc)
        return {
            "st": src_train_set, "sv": src_val_set,
            "tt": tgt_train_set, "tv": tgt_val_set
        }

    def get_data_loader(self, ds: dict):
        torch_loader = torch.utils.data.DataLoader
        bs = self.args.batch_size
        return {
            "st": torch_loader(ds["st"], batch_size=bs, shuffle=True),
            "sv": torch_loader(ds["sv"], batch_size=bs, shuffle=False),
            "tt": torch_loader(ds["tt"], batch_size=bs, shuffle=True),
            "tv": torch_loader(ds["tv"], batch_size=bs, shuffle=False)
        }

    def set_up(self):
        """
        Initialize: optimizer, loss_funcs, scheduler;
        :return:
        """
        self.model = self.get_model().to(self.device)
        self.optimizer = self.get_optimizer(self.model)
        self.scheduler = self.get_scheduler(self.optimizer, self.args.lr_scheduler_type)
        self.loader_dict = self.get_data_loader(self.get_dataset())

    def validate(self):
        self.model.eval()
        val_loss = x_tools.para_utils.AverageMeter()
        correct = 0
        criterion = nn.CrossEntropyLoss()
        target_loader = self.loader_dict["tv"]
        len_of_dataset = len(target_loader.dataset)
        with torch.no_grad():
            for data, label in target_loader:
                data, label = data.to(self.device), label.to(self.device)
                predict = self.model.predict(data)
                val_loss.update(criterion(predict, label).item())
                out_fault_type = torch.max(predict, 1)[1]
                correct += torch.sum(out_fault_type == label)
        # print("correct samples:{}".format(correct))
        acc = 100.*correct.item()/len_of_dataset
        return acc, val_loss.avg

    def train(self):
        """
        :return: execute after self.setup()
        """
        save_dir = "checkpoint/" + self.args.dataset + "_{}-{}_{}_{}".format(
            self.args.src_wc, self.args.tgt_wc, self.args.basic_net, self.args.transfer_loss
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.args.is_continue_train:
            self.load_checkpoint(save_dir + "/checkpoint.tar")

        len_of_loader = min(len(self.loader_dict["st"]), len(self.loader_dict["tt"])) if self.loader_dict else 0
        best_acc, best_epoch = 0., 0
        for e in tqdm(range(self.start_epoch, self.args.n_epoch)):
            self.model.train()
            train_loss_cls = x_tools.para_utils.AverageMeter()
            train_loss_transfer = x_tools.para_utils.AverageMeter()
            train_loss_total = x_tools.para_utils.AverageMeter()
            if len_of_loader:
                iter_st, iter_tt = iter(self.loader_dict["st"]), iter(self.loader_dict["tt"])
            transfer_loss_weight = 2 / (1 + math.exp(-10 * e / self.args.n_epoch)) - 1 if \
                                   self.args.is_descend else self.args.transfer_loss_weight
            for n in range(len_of_loader):
                [st_data, st_label] = [i.to(self.device) for i in next(iter_st)]
                [tt_data, tt_label] = [i.to(self.device) for i in next(iter_tt)]
                cls_loss, transfer_loss = self.model(st_data, tt_data, st_label)
                total_loss = 0.*cls_loss + transfer_loss_weight*transfer_loss               # #######################
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                train_loss_cls.update(cls_loss.item())
                train_loss_transfer.update(transfer_loss.item())
                train_loss_total.update(total_loss.item())
            info = "Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss: {:.4f}, total_Loss: {:.4f}".format(
                        e, self.args.n_epoch, train_loss_cls.avg, train_loss_transfer.avg, train_loss_total.avg)
            val_acc, val_loss = self.validate()
            info += ", val_loss {:.4f}, val_acc: {:.4f}".format(val_loss, val_acc)
            info += ", The current lr {:.6f}".format(self.optimizer.state_dict()['param_groups'][0]['lr'])
            print("\n" + info)
            if self.args.is_save_checkpoint:
                self.save_checkpoint(e, save_dir)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = e
                if self.args.is_save_checkpoint:
                    self.save_checkpoint(e, save_dir, filename="checkpoint_max.tar")
            if self.scheduler:
                self.scheduler.step()
        name = self.args.dataset + self.args.src_wc + "-" + self.args.tgt_wc
        line = name + ": Best accuracy {:.4f} got in epoch {} by {}".format(best_acc, best_epoch, self.args.basic_net) + "\n"
        print(line)
        with open("Best_acc.txt", "a", encoding="utf-8") as f:
            f.write(line)

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

    def train_source_net(self):
        model = self.get_model().source_net.to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=self.args.momentum, weight_decay=self.args.weight_decay,
                              nesterov=False)
        criterion = torch.nn.CrossEntropyLoss()
        steps = int(10)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, steps, self.args.gamma)
        EPOCH = 100
        len_of_loader = min(len(self.loader_dict["st"]), len(self.loader_dict["tt"])) if self.loader_dict else 0
        best_acc, best_epoch = 0., 0
        for e in range(EPOCH):
            model.train()
            train_loss = x_tools.para_utils.AverageMeter()
            if len_of_loader:
                iter_st = iter(self.loader_dict["st"])
            for n in range(len_of_loader):
                [st_data, st_label] = [i.to(self.device) for i in next(iter_st)]
                output = model(st_data)
                cls_loss = criterion(output, st_label)
                optimizer.zero_grad()
                cls_loss.backward()
                optimizer.step()
                train_loss.update(cls_loss.item())
            info = "EPOCH: [{:2d}/{}], cls_loss: {:4f}".format(e, EPOCH, train_loss.avg)
            if lr_scheduler:
                lr_scheduler.step()
            # test
            len_of_test_set = len(self.loader_dict["sv"])
            len_of_samples = len(self.loader_dict["sv"].dataset)
            model.eval()
            correct = 0
            with torch.no_grad():
                iter_sv = iter(self.loader_dict["sv"])
                for m in range(len_of_test_set):
                    [sv_data, sv_label] = [j.to(self.device) for j in next(iter_sv)]
                    predict = model(sv_data)
                    correct += torch.sum(torch.max(predict, 1)[1] == sv_label)
            acc = 100. * correct/len_of_samples
            info += " ACC: {:4f}".format(acc)
            info += " The current lr {:.6f}".format(optimizer.state_dict()['param_groups'][0]['lr'])
            print(info)
            best_acc = acc if best_acc < acc else acc
            # datn load pretrained weight:
            if best_acc > 99:
                # self.model.load_state_dict(model.state_dict())
                self.model.basic_net_s.load_state_dict(model[0].state_dict())
                self.model.basic_net_t.load_state_dict(model[0].state_dict())
                self.model.classifier.load_state_dict(model[1].state_dict())
                break








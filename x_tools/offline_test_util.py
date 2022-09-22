import torch, os, CNN_Datasets, networks_transfer, networks_basic, x_tools
import torch.nn as nn


class test_transfer:
    """
    Need: 1. the same args at the training time; self.set_up() AND self.acc_calculate() will get the acc and loss.
    2. The checkpoint tar file should be given too.
    """
    def __init__(self, args, checkpoint_filename):
        super(test_transfer, self).__init__()
        self.args = args
        self.checkpoint_filename = checkpoint_filename
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model, self.loader_dict = None, None

    def get_model(self):
        return networks_transfer.transfernet1d(num_class=self.args.num_class, basic_net=self.args.basic_net,
                                               transfer_loss=self.args.transfer_loss)

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

    def get_model_state_dict(self):
        checkpoint = torch.load(self.checkpoint_filename)
        return checkpoint["model_state_dict"]

    def set_up(self):
        self.model = self.get_model().to(self.device)
        self.model.load_state_dict(self.get_model_state_dict())
        self.model.eval()
        self.loader_dict = self.get_data_loader(self.get_dataset())

    def acc_calculate(self):
        val_loss = x_tools.para_utils.AverageMeter()
        correct = 0
        criterion = nn.CrossEntropyLoss()
        target_loader = self.loader_dict["tt"]
        len_of_dataset = len(target_loader.dataset)
        with torch.no_grad():
            for data, label in target_loader:
                data, label = data.to(self.device), label.to(self.device)
                predict = self.model.predict(data)
                val_loss.update(criterion(predict, label).item())
                out_fault_type = torch.max(predict, 1)[1]
                correct += torch.sum(out_fault_type == label)
        acc = 100. * correct / len_of_dataset
        return acc, val_loss.avg


class test_basic:
    """
    Need: 1. the same args at the training time; self.set_up() AND self.acc_calculate() will get the acc and loss.
    2. The checkpoint tar file should be given too.
    """
    def __init__(self, args, checkpoint_filename):
        super(test_basic, self).__init__()
        self.args = args
        self.checkpoint_filename = checkpoint_filename
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model, self.loader_dict = None, None

    def get_model(self):
        return getattr(networks_basic, self.args.basic_net)(out_channel=self.args.num_class, is_tl=False)

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

    def get_model_state_dict(self):
        checkpoint = torch.load(self.checkpoint_filename)
        return checkpoint["model_state_dict"]

    def set_up(self):
        self.model = self.get_model().to(self.device)
        self.model.load_state_dict(self.get_model_state_dict())
        self.model.eval()
        self.loader_dict = self.get_data_loader(self.get_dataset())

    def acc_calculate(self):
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
        acc = 100. * correct / len_of_dataset
        return acc, val_loss.avg


if __name__ == "__main__":
    from train_transfer import get_args
    args = get_args()
    args.data_dir = "../../../Dataset/CQDX/CQDXGear/gear_fault_rawdata_1024000x1_simple.mat"
    checkpoint_filename = "../checkpoint/CQDXGear1D_1-0_cnn1d_adv/checkpoint_max.tar"
    tester = test_transfer(args, checkpoint_filename=checkpoint_filename)
    tester.set_up()
    print(tester.acc_calculate())


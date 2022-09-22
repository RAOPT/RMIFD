import CNN_Datasets, networks_basic, torch, networks_transfer
from tqdm import tqdm
from torchvision import datasets, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_loader(dataset):
    return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)


def get_criterion(loss_func="CE"):
    if loss_func == "CE":
        return torch.nn.CrossEntropyLoss()
    elif loss_func == "BCE":
        return torch.nn.BCELoss()
    elif loss_func == "MSE":
        return torch.nn.MSELoss()


def train(model, train_loader, val_loader):
    criterion = get_criterion()
    for e in tqdm(range(50)):
        for i, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)
            # pred = model(data)
            pred = model(data, data, label)
            loss = criterion(pred, label)
    pass


if __name__ == "__main__":
    torch_dataset = CNN_Datasets.CQDXGear1D()
    train_dataset, val_dataset = torch_dataset.data_prepare()
    train_loader = get_loader(train_dataset)
    val_loader = get_loader(val_dataset)
    # basic model
    # model = networks_basic.cnn1d(out_channel=5, is_tl=True).to(device)
    # model = networks_basic.resnet1d(is_tl=True).to(device)
    # transfer_model
    model = networks_transfer.transfernet1d(num_class=5, basic_net="cnn1d", transfer_loss="adv").to(device)
    train(model, train_loader, val_loader)



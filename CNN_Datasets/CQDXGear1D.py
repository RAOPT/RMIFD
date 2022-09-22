import pandas as pd
from scipy.io import loadmat
from datasets_tools.SequenceDatasets import dataset
from datasets_tools.Sequence_Aug import *


mat_data_path = "E:/AAA/Dataset/CQDX/CQDXGear/gear_fault_rawdata_1024000x1_simple.mat"
class_list = ["normal", "surfacewear", "crackroot", "chipped", "missing"]
label_list = [0, 1, 2, 3, 4]


def get_data_from_file(root: str, wc: str, signal_size=1024):
    total_data, total_label = list(), list()
    raw_data = loadmat(root)
    for num, f_type in enumerate(class_list):
        f_data = raw_data["rawdata_1024000x1"][f_type][0][0][0, int(wc)]
        # raw_data["rawdata_1024000x1"][故障类型][0][0][0, 工况]
        data, label = list(), list()
        start, end = 0, signal_size
        while end <= f_data.shape[0]:
            data.append(f_data[start:end])
            label.append(label_list[num])
            start += signal_size
            end += signal_size
        total_data += data
        total_label += label
    return [total_data, total_label]


def train_test_split_order(data_pd, test_size, num_class=5):
    train_pd = pd.DataFrame(columns=('data', 'label'))
    val_pd = pd.DataFrame(columns=('data', 'label'))
    for i in range(num_class):
        data_pd_tmp = data_pd[data_pd['label'] == i].reset_index(drop=True)
        train_pd = train_pd.append(data_pd_tmp.loc[:int((1 - test_size) * data_pd_tmp.shape[0])-1, ['data', 'label']],
                                   ignore_index=True)
        val_pd = val_pd.append(data_pd_tmp.loc[int((1 - test_size) * data_pd_tmp.shape[0]):, ['data', 'label']],
                               ignore_index=True)
        # loc[0:10]取值到index=10而非9;
    return train_pd, val_pd


def data_transforms(dataset_type="train", normlize_type="0-1"):
    transforms = {
            'train': Compose([
            Reshape(),
            Normalize(normlize_type),
            RandomAddGaussian(),
            RandomScale(),
            RandomStretch(),
            RandomCrop(),
            Retype()
            ]),
            'val': Compose([
            Reshape(),
            Normalize(normlize_type),
            Retype()
            ])
            }
    return transforms[dataset_type]


class CQDXGear1D(object):
    num_class = 5
    input_channel = 1

    def __init__(self, data_dir=mat_data_path, normalize_type="0-1"):
        self.data_dir = data_dir
        self.normalize_type = normalize_type

    def data_prepare(self, wc="0"):
        """
        :param test, wc: for every 1-D dataset, "wc" parameter could choose different domains
        :return: torch dataset of 1-D vibration data with sample size=1024
        """
        total_data = get_data_from_file(self.data_dir, wc=wc)
        data_pd = pd.DataFrame({"data": total_data[0], "label": total_data[-1]})
        train_pd, val_pd = train_test_split_order(data_pd, 0.3, 5)
        train_dataset = dataset(train_pd, transform=data_transforms("train", self.normalize_type))
        val_dataset = dataset(val_pd, transform=data_transforms("val", self.normalize_type))
        # print("使用的0.3比例测试")
        return train_dataset, val_dataset


class CQDXGearFFT(object):
    num_class = 5
    input_channel = 1

    def __init__(self, data_dir=mat_data_path, normalize_type="0-1"):
        self.data_dir = data_dir
        self.normalize_type = normalize_type

    def data_prepare(self, wc="0"):
        """
        :param test, wc: for every 1-D dataset, "wc" parameter could choose different domains
        :return: torch dataset of 1-D vibration data with sample size=1024

        """
        total_data = get_data_from_file(self.data_dir, wc=wc, signal_size=2048)
        import numpy as np
        for n in range(len(total_data[0])):
            total_data[0][n] = np.fft.fft(total_data[0][n])
            total_data[0][n] = np.abs(total_data[0][n]) / len(total_data[0][n])
            total_data[0][n] = total_data[0][n][range(int(total_data[0][n].shape[0] / 2))]
            total_data[0][n] = total_data[0][n].reshape(-1, 1)
        data_pd = pd.DataFrame({"data": total_data[0], "label": total_data[-1]})
        train_pd, val_pd = train_test_split_order(data_pd, 0.1, 5)
        train_dataset = dataset(train_pd, transform=data_transforms("train", self.normalize_type))
        val_dataset = dataset(val_pd, transform=data_transforms("val", self.normalize_type))
        return train_dataset, val_dataset


class CQDXGear1DaFFT(object):
    num_class = 5
    input_channel = 1

    def __init__(self, data_dir=mat_data_path, normalize_type="0-1"):
        self.data_dir = data_dir
        self.normalize_type = normalize_type

    def data_prepare(self, wc="0"):
        """
        :param test, wc: for every 1-D dataset, "wc" parameter could choose different domains
        :return: torch dataset of 1-D vibration data with sample size=1024

        """
        total_data = get_data_from_file(self.data_dir, wc=wc, signal_size=2048)
        import numpy as np
        import copy
        backup_total_data = copy.deepcopy(total_data)
        for n in range(len(total_data[0])):
            total_data[0][n] = np.fft.fft(total_data[0][n])
            total_data[0][n] = np.abs(total_data[0][n]) / len(total_data[0][n])
            total_data[0][n] = total_data[0][n][range(int(total_data[0][n].shape[0] / 2))]
            total_data[0][n] = total_data[0][n].reshape(-1, 1)
            # merge
            total_data[0][n] = np.vstack((total_data[0][n], backup_total_data[0][n]))
        data_pd = pd.DataFrame({"data": total_data[0], "label": total_data[-1]})
        train_pd, val_pd = train_test_split_order(data_pd, 0.1, 5)
        train_dataset = dataset(train_pd, transform=data_transforms("train", self.normalize_type))
        val_dataset = dataset(val_pd, transform=data_transforms("val", self.normalize_type))
        return train_dataset, val_dataset


if __name__ == "__main__":
    CQUG = CQDXGear1D(mat_data_path, "0-1")
    CQUG.data_prepare()

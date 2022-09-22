import os
import warnings, copy
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from datasets_tools.SequenceDatasets import dataset
from datasets_tools.Sequence_Aug import *


mat_data_dir = "E:/AAA/Dataset/For_the_strange_Benchmark/CWRU"
class_list = ['normal_', 'B007_', 'B014_', 'B021_',
              'IR007_', 'IR014_', 'IR021_',
              'OR007@6_', 'OR014@6_', 'OR021@6_']
label_list = [i for i in range(10)]


def get_data_from_file(root: str, wc: str, direction=0, signal_size=1024):
    total_data, total_label = list(), list()
    for num, i in enumerate(class_list):
        raw_data = loadmat(os.path.join(root, i+wc+".mat"))
        if "DE" in list(raw_data.keys())[3]:
            # print(raw_data[list(raw_data)[3]].shape)
            # print(list(raw_data)[3])
            raw_data = raw_data[list(raw_data)[3]][:120832, :]
            # 驱动端数据
        # elif "DE" in list(raw_data.keys())[4]:
        #     raw_data = raw_data[list(raw_data)[4]][:112640, :]
        else:
            warnings.warn(message="No f***ing damn file.")
        data, label = list(), list()
        start, end = 0, signal_size
        while end <= len(raw_data):
            data.append(raw_data[start:end].reshape(-1, 1))
            label.append(label_list[num])
            start += signal_size
            end += signal_size
        total_data += data
        total_label += label
    return [total_data, total_label]


def train_test_split_order(data_pd, test_size, num_class=10):
    train_pd = pd.DataFrame(columns=("data", "label"))
    val_pd = pd.DataFrame(columns=("data", "label"))
    for i in range(num_class):
        data_pd_tmp = data_pd[data_pd['label'] == i].reset_index(drop=True)
        train_pd = train_pd.append(data_pd_tmp.loc[:int((1 - test_size) * data_pd_tmp.shape[0]) - 1, ['data', 'label']],
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
            # RandomAddGaussian(),
            # RandomScale(),
            # RandomStretch(),
            # RandomCrop(),
            Retype()
            ]),
            'val': Compose([
            Reshape(),
            Normalize(normlize_type),
            Retype()
            ])
            }
    return transforms[dataset_type]


class CWRUBear1D(object):
    num_class = 5
    input_channel = 1

    def __init__(self, data_dir=mat_data_dir, normalize_type="0-1"):
        self.data_dir = data_dir
        self.normalize_type = normalize_type

    def data_prepare(self, wc="0"):
        """
        :param test, wc: for every 1-D dataset, "wc" parameter could choose different domains
        :return: torch dataset of 1-D vibration data with sample size=1024(final)
        """
        total_data = get_data_from_file(self.data_dir, wc=wc)
        data_pd = pd.DataFrame({"data": total_data[0], "label": total_data[-1]})
        train_pd, val_pd = train_test_split_order(data_pd, 0.3, 10)
        train_dataset = dataset(train_pd, transform=data_transforms("train", self.normalize_type))
        val_dataset = dataset(val_pd, transform=data_transforms("val", self.normalize_type))
        return train_dataset, val_dataset


class CWRUBearFFT(object):
    num_class = 5
    input_channel = 1

    def __init__(self, data_dir=mat_data_dir, normalize_type="0-1"):
        self.data_dir = data_dir
        self.normalize_type = normalize_type

    def data_prepare(self, wc="0"):
        """
        :param test, wc: for every 1-D dataset, "wc" parameter could choose different domains
        :return: torch dataset of 1-D vibration data with sample size=1024

        """
        total_data = get_data_from_file(self.data_dir, wc=wc, signal_size=1024)

        for n in range(len(total_data[0])):
            total_data[0][n] = np.fft.fft(total_data[0][n])
            total_data[0][n] = np.abs(total_data[0][n]) / len(total_data[0][n])
            total_data[0][n] = total_data[0][n][range(int(total_data[0][n].shape[0] / 2))]
            total_data[0][n] = total_data[0][n].reshape(-1, 1)
        data_pd = pd.DataFrame({"data": total_data[0], "label": total_data[-1]})
        train_pd, val_pd = train_test_split_order(data_pd, 0.3, 10)
        train_dataset = dataset(train_pd, transform=data_transforms("train", self.normalize_type))
        val_dataset = dataset(val_pd, transform=data_transforms("val", self.normalize_type))
        return train_dataset, val_dataset


class CWRUBear1DaFFT(object):
    # num_class = 5
    # input_channel = 1

    def __init__(self, data_dir=mat_data_dir, normalize_type="0-1"):
        self.data_dir = data_dir
        self.normalize_type = normalize_type

    def data_prepare(self, wc="0"):
        """
        :param test, wc: for every 1-D dataset, "wc" parameter could choose different domains
        :return: torch dataset of 1-D vibration data with sample size=1024(final)
        """
        total_data = get_data_from_file(self.data_dir, wc=wc, signal_size=1024)
        backup_total_data = copy.deepcopy(total_data)
        for n in range(len(total_data[0])):
            total_data[0][n] = np.fft.fft(total_data[0][n])
            total_data[0][n] = np.abs(total_data[0][n]) / len(total_data[0][n])
            total_data[0][n] = total_data[0][n][range(int(total_data[0][n].shape[0] / 2))]
            total_data[0][n] = total_data[0][n].reshape(-1, 1)
            # 合并FFT与1D数据
            total_data[0][n] = np.vstack((total_data[0][n], backup_total_data[0][n]))

        data_pd = pd.DataFrame({"data": total_data[0], "label": total_data[-1]})
        train_pd, val_pd = train_test_split_order(data_pd, 0.3, 10)
        train_dataset = dataset(train_pd, transform=data_transforms("train", self.normalize_type))
        val_dataset = dataset(val_pd, transform=data_transforms("val", self.normalize_type))
        return train_dataset, val_dataset


if __name__ == "__main__":
    get_data_from_file(root=mat_data_dir, wc="2")
    CWRUB = CWRUBear1DaFFT(mat_data_dir, "0-1")
    CWRUB.data_prepare()

import torch.nn as nn


# -----------------------input size>=32---------------------------------
class lenet1d(nn.Module):
    def __init__(self, in_channel=1, out_channel=10, is_tl=False):
        super(lenet1d, self).__init__()
        self.is_tl = is_tl
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channel, 6, 5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),      # /2
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(6, 16, 5),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(5)  # adaptive change the output_size to (16,5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5, 30),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(30, 10),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(10, out_channel)

    def forward(self, x):
        x = self.conv1(x)
        feature = self.conv2(x)
        x = feature.view(feature.size()[0], -1)
        if self.is_tl:
            return x
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def out_dim(self):
        return self.fc1[0].in_features if self.is_tl else self.fc3.out_features

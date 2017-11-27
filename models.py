import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 28, kernel_size=5, padding=2),
            nn.BatchNorm2d(28),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(28, 28, kernel_size=5, padding=2),
            nn.BatchNorm2d(28),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(28, 28, kernel_size=3, padding=1),
            nn.BatchNorm2d(28),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*28, 64)
        self.fc2 = nn.Linear(64, 64)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.fc2(out)
        return F.log_softmax(out)

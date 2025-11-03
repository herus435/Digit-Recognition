import torch
from torch import nn
import torch.nn.functional as F

# Modelimizi tanımlıyoruz
class MYNet(nn.Module):
    def __init__(self):
        super(MYNet, self).__init__()
        self.conv = nn.Conv2d(1, 32, 3, padding=1)
        self.conv1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv(x), 2))
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

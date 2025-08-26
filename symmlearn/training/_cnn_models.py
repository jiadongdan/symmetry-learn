import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    A CNN for image classification with 6 input channels and 3 output classes.
    Includes BatchNorm and Global Average Pooling.
    """

    def __init__(self, dropout_rate=0.3, num_classes=3, num_input_channels=6):
        super(CNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.num_input_channels = num_input_channels

        # Conv backbone with BatchNorm
        self.conv1 = nn.Conv2d(num_input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)

        # Global Average Pooling (makes model input-size independent)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected head
        self.fc1 = nn.Linear(128, 64)   # Compact instead of huge flatten
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)

        # Activation
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pool(self.act(self.bn1(self.conv1(x))))
        x = self.pool(self.act(self.bn2(self.conv2(x))))
        x = self.pool(self.act(self.bn3(self.conv3(x))))

        x = self.gap(x)                  # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)        # (B, 128)

        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
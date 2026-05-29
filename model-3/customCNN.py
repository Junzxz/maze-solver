import torch.nn as nn

class customCNN(nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
        )

        self.fcn = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], 128, 16*16).mean(2)  # GAP
        x = self.fcn(x)
        return x
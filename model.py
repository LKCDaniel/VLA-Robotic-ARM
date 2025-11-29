import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)  # 加上这行
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.stage1 = ResidualBlock(16, 32)
        self.stage2 = ResidualBlock(32, 64)
        self.stage3 = ResidualBlock(64, 128)
        self.stage4 = ResidualBlock(128, 32)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # 初始层
        out = self.conv1(x)

        # 四个阶段
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        # 最终层
        out = self.pooling(out)
        out = torch.flatten(out, 1)

        return out


class BottleneckBlock(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.BatchNorm1d(dim2),
            nn.ReLU(inplace=True),
            nn.Linear(dim2, dim1),
            nn.BatchNorm1d(dim1)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.layers(x) + x
        out = self.relu(out)
        return out


class MLP(nn.Module):
    def __init__(self, in_dim, h_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([BottleneckBlock(in_dim, h_dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class VisionActionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ResNet()
        self.feature_projector = nn.Linear(32 * 2 + 5, 16)
        self.feature_fuser = MLP(in_dim=16, h_dim=32, num_layers=2)
        self.action_predictor = nn.Sequential(nn.Linear(16, 3), nn.Tanh())
        self.state_predictor = nn.Linear(16, 2)

    def forward(self, img1, img2, state):
        f1 = self.image_encoder(img1)
        f2 = self.image_encoder(img2)
        f = torch.concatenate([f1, f2, state], dim=-1)
        f = self.feature_projector(f)
        f = self.feature_fuser(f)
        action = self.action_predictor(f)
        next_state = self.state_predictor(f)
        out = torch.concatenate([action, next_state], dim=-1)
        return out

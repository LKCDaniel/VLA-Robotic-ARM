import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)

    def forward(self, x):
        # 主路径前向传播
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接 + 最终激活
        out += self.shortcut(x)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=6, stride=2, padding=3, bias=False)
        self.stage1 = ResidualBlock(32, 64)
        self.stage2 = ResidualBlock(64, 128)
        self.stage3 = ResidualBlock(128, 256)
        self.stage4 = ResidualBlock(256, 32)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        self.image_encoder1 = ResNet()
        self.image_encoder2 = ResNet()
        self.feature_projector = nn.Linear(69, 16)
        self.feature_fuser = MLP(in_dim=16, h_dim=64, num_layers=4)
        self.action_predictor = nn.Sequential(nn.Linear(16, 3), nn.Tanh())
        self.catch_state_predictor = nn.Sequential(nn.Linear(16, 1), nn.Softmax(dim=-1))
        self.task_state_predictor = nn.Sequential(nn.Linear(16, 1), nn.Softmax(dim=-1))

    def forward(self, img1, img2, state):
        f1 = self.image_encoder1(img1)
        f2 = self.image_encoder2(img2)
        f = torch.concatenate([f1, f2, state], dim=-1)
        f = self.feature_projector(f)
        f = self.feature_fuser(f)
        action = self.action_predictor(f)
        catch_state = self.catch_state_predictor(f)
        task_state = self.task_state_predictor(f)
        return action, catch_state, task_state


def main1():
    device = "cuda"
    model = ResNet()
    model = model.to(device)
    input_tensor = torch.randn(1, 3, 512, 512).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        print(output.shape)


def main2():
    device = "cuda"
    model = MLP(512, 32, 5, 4)
    model = model.to(device)
    input_tensor = torch.randn(13, 512).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        print(output.shape)


if __name__ == '__main__':
    device = "cuda"
    model = VisionActionModel()
    model = model.to(device)
    img1 = torch.randn(13, 3, 512, 512).to(device)
    img2 = torch.randn(13, 3, 512, 512).to(device)
    state = torch.randn(13, 5).to(device)
    action, catch_state, task_state = model(img1, img2, state)
    print(action.shape, catch_state.shape, task_state.shape)
    print(catch_state, task_state)

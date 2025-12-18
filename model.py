import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import resnet18, ResNet18_Weights


class VLA_transformer(nn.Module):
    def __init__(self, state_dim=3, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        # resnet_state_dict = load_file("resnet_18_pretrained.safetensors")
        # resnet.load_state_dict(resnet_state_dict)
        img_backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove the final fully connected layer
        for param in img_backbone.parameters():
            param.requires_grad = False
    
        self.img_encoder = nn.Sequential(
            img_backbone,
            nn.Flatten(),
            nn.Linear(512, d_model, bias=False)
        )
        self.state_encoder = nn.Linear(state_dim, d_model, bias=False)
        self.pos_embed = nn.Parameter(torch.randn(1, 4, d_model)*0.02)  # 4 tokens: 3 images + 1 state
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            norm_first=True,
            dim_feedforward=d_model*4,
            batch_first=True,
            activation='gelu',
            bias=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        
        self.layer_norm = nn.LayerNorm(d_model, bias=False)
        self.head = nn.Linear(4*d_model, 4, bias=True)  # 3 for action prediction, 1 for task completion logits
        # self.logits_head = nn.Linear(4*d_model, 1, bias=False)
        
    def forward(self, img1, img2, img3, state): # img: (N, C, H, W), state: (4)
        batch_size = img1.size(0)
        
        imgs = torch.stack([img1, img2, img3], dim=1).view(batch_size*3, *img1.shape[1:])  # (N*3, C, H, W)
        img_tokens = self.img_encoder(imgs).view(batch_size, 3, -1)  # (N, 3, d_model)
        state_token = self.state_encoder(state).unsqueeze(1)  # (N, 1, d_model)
        
        context = torch.cat([img_tokens, state_token], dim=1) + self.pos_embed  # (N, 4, d_model)
        output = self.transformer(context)  # (N, 4, d_model)
        output = self.layer_norm(output).view(batch_size, -1)  # (N, 4*d_model)
        output = self.head(output)  # (N, 4)
        action, logits = output[:, :3], output[:, 3:] # (N, 3), (N, 1)
        
        return action, logits


# Homoscedastic uncertainty weighting for multi-task learning
class AutomaticWeightedLoss(nn.Module):
    def __init__(self, *init_scales, num_tasks=2):
        super().__init__()
        # Initialize "s" (log variance) to 0.0. 
        # s = log(sigma^2). Use s instead of sigma for numerical stability.
        if init_scales and len(init_scales) == num_tasks:
            init_s = [-math.log(w) for w in init_scales]
            self.params = nn.Parameter(torch.tensor(init_s, dtype=torch.float32))
        else:
            self.params = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, action_loss, task_loss):
        # --- Task 0: Action (Regression) ---
        # Formula: (1 / 2*exp(s)) * Loss + s/2
        s_action = self.params[0]
        # We multiply by 0.5 for regression tasks (Gaussian assumption)
        weighted_action_loss = 0.5 * (torch.exp(-s_action) * action_loss + s_action)

        # --- Task 1: Task Completion (Classification) ---
        # Formula: (1 / exp(s)) * Loss + s/2 (approx for classification)
        s_task = self.params[1]
        weighted_task_loss = torch.exp(-s_task) * task_loss + 0.5 * s_task

        return weighted_action_loss + weighted_task_loss
        
    















# legacy below

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
        self.conv1 = nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.stage1 = ResidualBlock(8, 16)
        self.stage2 = ResidualBlock(16, 32)
        self.stage3 = ResidualBlock(32, 64)
        self.stage4 = ResidualBlock(64, 32)
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
        d_model = 16
        self.image_encoder = ResNet()
        self.feature_projector = nn.Linear(32 * 3 + 4, d_model)
        self.feature_fuser = MLP(in_dim=d_model, h_dim=d_model*2, num_layers=2)
        self.position_delta_predictor = nn.Sequential(nn.Linear(d_model, 3), nn.Tanh())
        self.next_catch_predictor = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.next_task_predictor = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, img1, img2, img3, state):
        f1 = self.image_encoder(img1)
        f2 = self.image_encoder(img2)
        f3 = self.image_encoder(img3)
        f = torch.concatenate([f1, f2, f3, state], dim=-1)
        f = self.feature_projector(f)
        f = self.feature_fuser(f)
        position_delta = self.position_delta_predictor(f)
        # position_delta = F.normalize(position_delta, p=2, dim=1) # don't normalize
        # next_catch = self.next_catch_predictor(f)
        next_task = self.next_task_predictor(f)
        next_state = torch.concatenate([position_delta, next_task], dim=-1)
        return next_state

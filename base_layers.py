import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_wavelets import DWTForward, DWTInvers
from pytorch_wavelets import DWTForward, DWTInverse
from torchvision.ops import deform_conv2d
class Flatten(nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)
class CAM(nn.Module):
    def __init__(self,in_channels,reduction_ratio):
        super(CAM, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.Softsign(),
            nn.Linear(in_channels // reduction_ratio, in_channels ),
            nn.Softsign()
            )
    def forward(self,input):
        return input* self.module(input).unsqueeze(2).unsqueeze(3).expand_as(input)


class MIDA(nn.Module):
    def __init__(self, filters, activation='lrelu', save_dir='feature_maps'):
        super().__init__()
        # 分支1：1×1卷积（捕获点特征）
        self.branch1 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        # 分支2：1×1降维 + 3×3卷积（捕获中等尺度）
        self.branch2 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        # 分支3：1×1降维 + 5×5卷积（捕获大尺度）
        self.branch3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=5, padding=2),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        self.cam1 = CAM(filters, 4)

        self.cam2 = CAM(filters, 4)
        self.cam3 = CAM(filters, 4)
        self.branch3_up1 = nn.ConvTranspose2d(filters, filters, kernel_size=4, stride=2, padding=1)
        self.branch3_up2 = nn.ConvTranspose2d(filters, filters, kernel_size=4, stride=2, padding=1)
        # 融合层
        self.fusion = nn.Conv2d(filters * 4, filters, kernel_size=1)

    def forward(self, R):
        R_att = R
        feat1 = self.branch1(R_att)
        feat1_z = feat1
        feat1 = self.cam1(feat1_z)
        feat2 = self.branch2(R_att)
        feat2_1 = feat2
        feat2_z = feat2 - feat1_z
        feat2 = self.cam2(feat2_z)
        feat3 = self.branch3(R_att)
        feat3_z = feat3 -feat1_z-feat2_1
        feat3 = self.cam3(feat3_z)
        concat = torch.cat([R, feat1, feat2, feat3], dim=1)
        out = self.fusion(concat)
        return out

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, activation='lrelu', stride=1):
        super().__init__()
        self.ActivationLayer = nn.LeakyReLU(inplace=True)
        if activation == 'relu':
            self.ActivationLayer = nn.ReLU(inplace=True)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            self.ActivationLayer,
        )

    def forward(self, x):
        return self.conv_relu(x)


class ConvTranspose2D(nn.Module):
    def __init__(self, in_channels, out_channels, activation='lrelu'):
        super().__init__()
        self.ActivationLayer = nn.LeakyReLU(inplace=True)
        if activation == 'relu':
            self.ActivationLayer = nn.ReLU(inplace=True)
        self.deconv_relu = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0,output_padding=1 ),
            self.ActivationLayer,
        )

    def forward(self, x):
        return self.deconv_relu(x)

class Concat(nn.Module):
    def forward(self, x, y):
        _, _, xh, xw = x.size()
        _, _, yh, yw = y.size()
        diffY = xh - yh
        diffX = xw - yw
        y = F.pad(y, (diffX // 2, diffX - diffX//2,
                      diffY // 2, diffY - diffY//2))
        return torch.cat((x, y), dim=1)
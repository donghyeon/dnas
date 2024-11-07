import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from nni.mutable import label_scope
from nni.nas.nn.pytorch import LayerChoice, ModelSpace, MutableModule


class Shortcut(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__()
        self.use_identity = (in_ch == out_ch) and (stride == 1)
        if not self.use_identity:
            self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=stride, bias=False)
            self.bn = nn.BatchNorm2d(out_ch)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_identity:
            x = F.relu(self.bn(self.conv(x)))
        return x


class FBNetUnit(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, exp: int, kernel: int, stride: int, group: int) -> None:
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=exp*in_ch, kernel_size=1, groups=group)
        self.conv2 = nn.Conv2d(in_channels=exp*in_ch, out_channels=exp*in_ch, kernel_size=kernel, stride=stride, padding=int((kernel - 1)/2), groups=exp*in_ch)    # padding formula for when kernel is 3 or 5
        self.conv3 = nn.Conv2d(in_channels=exp*in_ch, out_channels=out_ch, kernel_size=1, groups=group)
        self.bn1 = nn.BatchNorm2d(exp*in_ch)
        self.bn2 = nn.BatchNorm2d(exp*in_ch)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.shortcut = Shortcut(in_ch=in_ch, out_ch=out_ch, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_tensor = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x))) + self.shortcut(input_tensor)
        return x


class TBS(MutableModule):
    def __init__(self, in_ch: int, out_ch: int, stride: int, num: int) -> None:
        super().__init__()
        self.num = num
        with label_scope():
            with label_scope("unit"):
                self.unit1 = LayerChoice([
                    FBNetUnit(in_ch=in_ch, out_ch=out_ch, exp=1, kernel=3, stride=stride, group=1),  # 0
                    FBNetUnit(in_ch=in_ch, out_ch=out_ch, exp=1, kernel=3, stride=stride, group=2),  # 1
                    FBNetUnit(in_ch=in_ch, out_ch=out_ch, exp=3, kernel=3, stride=stride, group=1),  # 2
                    FBNetUnit(in_ch=in_ch, out_ch=out_ch, exp=6, kernel=3, stride=stride, group=1),  # 3
                    FBNetUnit(in_ch=in_ch, out_ch=out_ch, exp=1, kernel=5, stride=stride, group=1),  # 4
                    FBNetUnit(in_ch=in_ch, out_ch=out_ch, exp=1, kernel=5, stride=stride, group=2),  # 5
                    FBNetUnit(in_ch=in_ch, out_ch=out_ch, exp=3, kernel=5, stride=stride, group=1),  # 6
                    FBNetUnit(in_ch=in_ch, out_ch=out_ch, exp=6, kernel=5, stride=stride, group=1),  # 7
                    Shortcut(in_ch=in_ch, out_ch=out_ch, stride=stride), # 8
                ])

                for i in range(1, num):
                    self.__setattr__(f"unit{i+1}", LayerChoice([
                        FBNetUnit(in_ch=out_ch, out_ch=out_ch, exp=1, kernel=3, stride=1, group=1),
                        FBNetUnit(in_ch=out_ch, out_ch=out_ch, exp=1, kernel=3, stride=1, group=2),
                        FBNetUnit(in_ch=out_ch, out_ch=out_ch, exp=3, kernel=3, stride=1, group=1),
                        FBNetUnit(in_ch=out_ch, out_ch=out_ch, exp=6, kernel=3, stride=1, group=1),
                        FBNetUnit(in_ch=out_ch, out_ch=out_ch, exp=1, kernel=5, stride=1, group=1),
                        FBNetUnit(in_ch=out_ch, out_ch=out_ch, exp=1, kernel=5, stride=1, group=2),
                        FBNetUnit(in_ch=out_ch, out_ch=out_ch, exp=3, kernel=5, stride=1, group=1),
                        FBNetUnit(in_ch=out_ch, out_ch=out_ch, exp=6, kernel=5, stride=1, group=1),
                        Shortcut(in_ch=out_ch, out_ch=out_ch, stride=1),
                    ]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unit1(x)
        for i in range(1, self.num):
            x = self.__getattr__(f"unit{i+1}")(x)
        return x


class FBNetSpace(ModelSpace):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        with label_scope("tbs"):
            self.tbs1 = TBS(in_ch=16, out_ch=16, stride=1, num=1)
            self.tbs2 = TBS(in_ch=16, out_ch=24, stride=2, num=4)
            self.tbs3 = TBS(in_ch=24, out_ch=32, stride=2, num=4)
            self.tbs4 = TBS(in_ch=32, out_ch=64, stride=2, num=4)
            self.tbs5 = TBS(in_ch=64, out_ch=112, stride=1, num=4)
            self.tbs6 = TBS(in_ch=112, out_ch=184, stride=2, num=4)
            self.tbs7 = TBS(in_ch=184, out_ch=352, stride=1, num=1)
        
        last_channel = 1504 # fbnet-a
        # last_channel = 1984 # fbnet-b,c
        self.conv2 = nn.Conv2d(in_channels=352, out_channels=last_channel, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(last_channel)
        self.head = nn.Conv2d(in_channels=last_channel, out_channels=num_classes, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias, 0.0)
        
        # # Zero gamma for the last bn in bottlenecks
        # for m in self.modules():
        #     if isinstance(m, FBNetUnit):
        #         init.constant_(m.bn3.weight, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.tbs1(x)
        x = self.tbs2(x)
        x = self.tbs3(x)
        x = self.tbs4(x)
        x = self.tbs5(x)
        x = self.tbs6(x)
        x = self.tbs7(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        return x


class FBNet(FBNetSpace):
    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.tbs1(x)
        x = self.tbs2(x)
        x = self.tbs3(x)
        x = self.tbs4(x)
        x = self.tbs5(x)
        x = self.tbs6(x)
        x = self.tbs7(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        return x

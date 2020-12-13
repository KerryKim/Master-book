import torch
import torch.nn as nn

##
'''
CBR2d를 함수가 아닌 클래스로 정의해 준다.
왜냐하면 nn.Module을 상속받고 있는 UNet 클래스에서 추출해 왔기 때문에
똑같은 hierarchy를 유지하기 위해 CBR2d도 nn.Module을 상속받을 수 있게 클래스로 바꾼다.
'''

class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm", relu=0.0):
        super().__init__()

        layers = []
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding,
                             bias=bias)]

        if not norm is None:
            if norm == "bnorm":
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == "inorm":
                layers += [nn.InstanceNorm2d(enumerate=out_channels)]

        if not relu is None:
            layers += [nn.ReLU() if relu == 0.0 else nn.LeakyReLU(relu)]

        # Sequential함수는 list로 만들어 놓은 여러 레이어를 한 번에 묶을 때 사
        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)

##

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 bias=True, norm="bnorm", relu=0.0):
        super().__init__()

        layers = []

        # 1st CBR2d
        layers += [CBR2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, bias=bias, norm=norm, relu=relu)]

        # 2st CBR2d
        # padding은 stride 사이즈와 동일한 1로 한다.
        layers += [CBR2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, bias=bias, norm=norm, relu=relu)]

        self.resblk = nn.Sequential(*layers)

    def forward(self, x):
        # SRResnet 논문에서 Elementwise Sum을 구현해주기 위해 return시 더해준다.
        return x + self.resblk(x)

##
'''
sub-pixel super resolution
https://arxiv.org/abs/1609.05158
'''

# Multi layers → one layer
# y방향의 down_sampling ratio = ry, x방향의 down_sampling ratio = rx로 정의

class PixelShuffle(nn.Module):
    def __init__(self, ry=2, rx=2):
        super().__init__()
        self.ry = ry
        self.rx = rx

    def forward(self, x):
        ry = self.ry
        rx = self.rx

        [B, C, H, W] = list(x.shape)

        x = x.reshape(B, C // (ry * rx), ry, rx, H, W)

        x = x.permute(0, 1, 4, 2, 5, 3)

        x = x.reshape(B, C // (ry * rx), H * ry, W * rx)

        return x


##
# one layer → multi layers
class PixelUnshuffle(nn.Module):
    def __init__(self, ry=2, rx=2):
        super().__init__()
        self.ry = ry
        self.rx = rx

    def forward(self, x):
        ry = self.ry
        rx = self.rx

        # batch size, n of channels, height, width
        [B, C, H, W] = list(x.shape)

        x = x.reshape(B, C, H // ry, ry, W // rx, rx)

        # permute는 axis의 축 순서를 바꿔주는 함수
        # 채널 뒤에 down sampling ratio를 먼저 써준다.
        x = x.permute(0, 1, 3, 5, 2, 4)

        # 1,3,5 axis에 해당하는 array가 채널 방향으로 쌓인다.
        x = x.reshape(B, C * (ry * rx), H // ry, W // rx)

        return x










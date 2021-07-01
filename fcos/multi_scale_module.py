if __name__ == '__main__':
    import sys
    assert False, "modify the following path to yours!"
    sys.path.append('/path/to/this/project')
import torch
from torch import nn


class RFALayer(nn.Module):
    def __init__(self, cfg, in_channels, stride=1 ,L=32, single_conv = False):
        super(RFALayer, self).__init__()
        self.branches = cfg.MODEL.SKC.BRANCH
        self.groups = cfg.MODEL.SKC.GROUP
        self.down_ratio = cfg.MODEL.SKC.DOWN_RATIO
        self.in_channels = in_channels
        self.d = max(int(in_channels/self.down_ratio), L)
        self.convs = nn.ModuleList([])
        if cfg.MODEL.SKC.DILATION:#use dilation convolution to increase RF
            for i in range(self.branches):
                self.convs.append(nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, dilation=1+i, padding=1+i, groups=self.groups),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=False)
                ))
        else:#increase RF by increasing the kernel size
            for i in range(branches):
                self.convs.append(nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3+2*i, stride=stride, padding=1+i, groups=self.groups),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=False)
                ))
        if single_conv:#envolving 1*1 conv
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=stride, padding=0, groups=self.groups),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=False)
            ))
            self.branches += 1
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
                nn.Linear(in_channels*self.branches, self.d),
                nn.ReLU(inplace = False),
                nn.Linear(self.d, in_channels*self.branches),
                nn.Sigmoid()
            )

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            feature = conv(x)
            if i == 0:
                features = feature
            else:
                features = torch.cat([features, feature], dim=1)
        feature_gp = self.gap(features).squeeze_()#batch*in_channels
        channel_weight = self.fc(feature_gp).unsqueeze(-1).unsqueeze(-1)
        return features * channel_weight

class RFAUnit(nn.Module):
    def __init__(self, cfg, in_channels, out_channels, mid_channels=None, stride=1, L=32, single_conv = False):
        super(RFAUnit, self).__init__()
        if single_conv:
            self.branches = cfg.MODEL.SKC.BRANCH+1
        else:
            self.branches = cfg.MODEL.SKC.BRANCH
        if mid_channels is None:
            mid_channels = int(in_channels/2)
        self.rfaunit = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(mid_channels),
            RFALayer(cfg, mid_channels, stride=stride, L=L, single_conv=single_conv),
            nn.BatchNorm2d(mid_channels * self.branches),
            nn.Conv2d(mid_channels * self.branches, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        feature = self.rfaunit(x)
        return feature



class SeLayer(nn.Module):
    def __init__(self, cfg, in_channels, L = 16):
        super(SeLayer, self).__init__()
        self.down_ratio = cfg.MODEL.SE.DOWN_RATIO
        #self.down_ratio = 4
        self.d = max(int(in_channels/self.down_ratio), L)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, self.d, bias = False),
            nn.ReLU(inplace = False),
            nn.Linear(self.d, in_channels, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channel, _, _ = x.size()
        y = self.gap(x).view(batch, channel)
        y = self.fc(y).view(batch, channel, 1, 1)
        #return x * y.expand_as(x)
        return y

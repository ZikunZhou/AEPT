
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import Backbone

def conv2d_dw_group(x, kernel):
    batch, channel = kernel.shape[:2]
    x = x.view(1, batch*channel, x.size(2), x.size(3))  # 1 * (b*c) * H * W
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))  # (b*c) * 1 * k * k
    out = F.relu(F.conv2d(x, kernel, groups=batch*channel))
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out

class DepthCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.fuse = nn.Sequential(
                nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                )

    def _init_weights(self):
        """
        function: Initialize model's parameters
        """
        for m in self.modules():

            if isinstance(m,nn.Conv2d):
                nn.init.normal_(m.weight.data,0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward_corr(self, kernel, input):
        kernel = self.conv_kernel(kernel)
        input = self.conv_search(input)
        feature = conv2d_dw_group(input, kernel)
        return feature

    def forward(self, kernel, search):
        corr_feature = self.forward_corr(kernel, search)
        siam_feature = self.fuse(corr_feature)
        return siam_feature

class Siamese(nn.Module):
    def __init__(self, bb_out_channels = [1024,], in_channels=256, hidden=256, out_channels=256):
        super(Siamese, self).__init__()
        self.out_channels = out_channels

        for (i,bb_out_channel) in enumerate(bb_out_channels):
            module_name = 'downsample{}'.format(i)
            self.add_module(module_name, nn.Sequential(
                        nn.Conv2d(bb_out_channel, in_channels, 1, bias=False),
                        nn.BatchNorm2d(in_channels),
                ))

        self.loc_cls = DepthCorr(in_channels, hidden, out_channels, kernel_size=3)

    def forward(self, ref_features, srch_features, cfg):
        """
        function: forward pass if siamese network when training
        args:
            ref_features (tuple[Tensor]) - len(tuple) = len(OUT_INDICES), Tensor.dim() = 5
            srch_features (tuple[Tensor]) - len(tuple) = len(OUT_INDICES), Tensor.dim() = 5
        returns:
            siam_features (list[Tensor]) - len(list) = len(OUT_INDICES) Tensor.dim() = 5

        """
        assert ref_features[0].dim() == 5, 'Expect 5  dimensional ref_features'
        num_stage = len(ref_features)
        crop_size = cfg.MODEL.SIAM.CROP_SIZE

        ds_ref_feats = [getattr(self, 'downsample{}'.format(i))(ref_feature[0,...]) for (i, ref_feature) in enumerate(ref_features)]
        ds_ref_feats = [ds_ref_feat[:,:,c:-c,c:-c] for (c, ds_ref_feat) in zip(crop_size, ds_ref_feats)]

        num_test_frames = srch_features[0].size(0)
        siam_features = []
        for i in range(num_stage):
            siam_features.append([self.loc_cls(ds_ref_feats[i],getattr(self, 'downsample{}'.format(i))(srch_feat)) for srch_feat in srch_features[i]])
        siam_features = [torch.stack(siam_feature, dim = 0) for siam_feature in siam_features]

        return siam_features

class xSiamese(nn.Module):
    def __init__(self, bb_out_channels = [1024,], in_channels=256, hidden=256, out_channels=256):
        super(xSiamese, self).__init__()
        self.out_channels = out_channels

        for (i,bb_out_channel) in enumerate(bb_out_channels):
            module_name = 'downsample{}'.format(i)
            self.add_module(module_name, nn.Sequential(
                        nn.Conv2d(bb_out_channel, in_channels, 1, bias=False),
                        nn.BatchNorm2d(in_channels),
                ))

        self.loc_cls = DepthCorr(in_channels, hidden, out_channels, kernel_size=3)


    def forward(self, srch_features, cfg):
        """
        function: forward pass if siamese network when testing
        args:
            srch_features (tuple[Tensor]) - len(tuple) = len(OUT_INDICES), Tensor.dim() = 5
            cfg
        """
        assert srch_features[0].dim() == 5, 'Expect 5 dimensional srch_features'
        self.srch_features = srch_features
        num_test_frames = srch_features[0].size(0)
        assert num_test_frames == 1, 'Expect 1 test frames for now'
        siam_features = []
        for i in range(self.num_stage):
            siam_features.append([self.loc_cls(self.ref_features[i],getattr(self, 'downsample{}'.format(i))(srch_feat)) for srch_feat in srch_features[i]])
        siam_features = [torch.stack(siam_feature, dim = 0) for siam_feature in siam_features]
        self.siam_features = siam_features

        return siam_features

    def temple(self,ref_features, cfg):
        """
        function: downsample and crop the ref_features
        args:
            ref_features (tuple[Tensor]) - len(tuple) = len(OUT_INDICES), Tensor.dim() = 5
        returns(not acturally return):
            self.ref_features list[Tensor] - len(list) = len(OUT_INDICES), Tensor.dim() = 4, cause num_train_images is always 1
        """
        assert ref_features[0].dim() == 5, 'Expect 5 dimensional ref_features'
        self.num_stage = len(ref_features)
        crop_size = cfg.MODEL.SIAM.CROP_SIZE
        ds_ref_feats = [getattr(self, 'downsample{}'.format(i))(ref_feature[0,...]) for (i, ref_feature) in enumerate(ref_features)]
        self.non_crop_ref_features = ds_ref_feats
        ds_ref_feats = [ds_ref_feat[:,:,c:-c,c:-c] for (c, ds_ref_feat) in zip(crop_size, ds_ref_feats)]
        self.ref_features = ds_ref_feats

def build_siamese(bb_out_channels = [1024,],in_channels=256, hidden=256, out_channels=256):
    return Siamese(bb_out_channels, in_channels, hidden, out_channels)

def build_xsiamese(bb_out_channels = [1024,],in_channels=256, hidden=256, out_channels=256):
    return xSiamese(bb_out_channels, in_channels, hidden, out_channels)

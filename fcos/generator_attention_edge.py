import torch
from torch import nn
import numpy as np
from torchvision.utils import make_grid
import cv2

from backbone import build_resnet50
from .siamese import build_siamese, build_xsiamese
from .fcos_attention_edge import build_fcos
from structures.bounding_box import BoxList

from matplotlib import pyplot as plt# for debug
from matplotlib.patches import Rectangle# for debug

class GenTracker(nn.Module):
    """
    Note: if we want to modift the structure of xsiamese network, we also need to modify the interface of self.siamese and self.head
          keep the structure of siamese and xsiamese the same
    """
    def __init__(self, cfg):
        super(GenTracker, self).__init__()

        self.config = cfg
        self.backbone = build_resnet50(cfg, init_weights = False)
        self.siamese = build_xsiamese(bb_out_channels = self.backbone.out_channels,
                                      in_channels=256,
                                      hidden=256,
                                      out_channels=256)
        self.fcos = build_fcos(cfg, self.siamese.out_channels, out_channels = 256)

    def forward(self, srch_images, image_size, tracking_state):
        """
        function: perform tracking by detection in srch_images
        args:
            srch_images (Tensor) - shape = [1,1,3,255,255]
            image_size (numpy.ndarray) - the size of the tracking image, not search window
                                         coordinate definition: [height, width]
            tracking_state (dict) - including the following keys:
                                        image_w (numpy.int?)- width of the tracking image
                                        image_h (numpy.int?)- height of the tracking image
                                        avg_chans (numpy.ndarray) - the average pixel value of each channel
                                        target_pos (numpy.ndarray) - the position of the object in last frame, coordinate definition[x_c, y_c]
                                        target_size (numpy.ndarray) - the size of the object in last frame, coordinate definition[width, height]
                                        scale_template (numpy.float64) - the scaling factor of the exemplar compared to object
                                        window (numpy.array) - penalty window
                                        tracking_config - the configurations of the tracker
                                        score - the
        """
        self.srch_images = srch_images# for debug
        if self.training:
            raise ValueError("Should perform tracking in GenTracker's eval mode")
        batch = srch_images.shape[-4]
        num_srch_images = srch_images.shape[0] if srch_images.dim() == 5 else 1

        srch_features = self.backbone(srch_images.view(-1, srch_images.shape[-3],srch_images.shape[-2], srch_images.shape[-1]))
        srch_features = [srch_feat.view(num_srch_images, batch, srch_feat.shape[-3], srch_feat.shape[-2], srch_feat.shape[-1])
                        for srch_feat in srch_features]
        siam_features = self.siamese(srch_features, self.config)
        target_pos, target_size, score = self.fcos(image_size, siam_features, tracking_state = tracking_state)
        return target_pos, target_size, score

    def temple(self, ref_images):
        """
        function: calculate, downsample and crop the ref_features from ref_images
        args:
            ref_images (Tensor) - shape = [1,1,3,127,127]
        """
        self.ref_images = ref_images# for debug
        if self.training:
            raise ValueError("Should perform tracking in GenTracker's eval mode")
        batch = ref_images.shape[-4]
        num_ref_images = ref_images.shape[0] if ref_images.dim() == 5 else 1 #for now, always 1
        ref_features = self.backbone(ref_images.view(-1, ref_images.shape[-3], ref_images.shape[-2], ref_images.shape[-1]))
        ref_features = [ref_feat.view(num_ref_images, batch, ref_feat.shape[-3], ref_feat.shape[-2], ref_feat.shape[-1])
                        for ref_feat in ref_features]

        self.siamese.temple(ref_features, self.config)

    def debug(self):
        """
        function:
            returns:
                ref_features (list[Tensor])
                non_crop_ref_features (list[Tensor])
                srch_features (list[Tnesor])
                siam_features (list[Tensor])
        """
        ref_features = [feat.clone().detach().to('cpu') for feat in self.siamese.ref_features]
        non_crop_ref_features = [feat.clone().detach().to('cpu') for feat in self.siamese.non_crop_ref_features]
        print(ref_features[0].requires_grad)
        tmp_srch_features = [feat.clone() for feat in self.siamese.srch_features]
        print(ref_features[0].shape)
        srch_features = []
        for i in range(len(tmp_srch_features)):
            srch_features.append(torch.stack([(self.siamese.downsample0(srch_feat)).detach().to('cpu') for srch_feat in tmp_srch_features[i]],dim = 0))
        print(srch_features[0].shape)
        siam_features = [feat.clone().detach().to('cpu') for feat in self.siamese.siam_features]
        print(siam_features[0].shape)
        return ref_features, non_crop_ref_features, srch_features, siam_features


def unnormal(tensor):
    return (((tensor.cpu().squeeze(0).numpy().transpose((1,2,0))+np.array([0.229,0.224,0.225]))*np.array([0.485,0.456,0.406]))*255+20).clip(0,255).astype(np.uint8)

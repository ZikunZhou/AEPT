from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet import models

class Backbone(nn.Module):
    def __init__(self, net_config, init_weights=True):
        super(Backbone,self).__init__()

        self.net_cfg = dict(
                type = net_config.TYPE,
                depth = net_config.DEPTH,
                num_stages = net_config.NUM_STAGES,
                strides = net_config.STRIDES,
                dilations = net_config.DILATIONS,
                out_indices = net_config.OUT_INDICES,
                frozen_stages = net_config.FROZEN_STAGES,
                dcn=dict(
                    modulated = net_config.DCN.MODULATED,
                    deformable_groups = net_config.DCN.DEFORMABLE_GROUPS,
                    fallback_on_stride = net_config.DCN.FALLBACK_ON_STRIDE
                    ),
                stage_with_dcn = net_config.STAGE_WITH_DCN
                )
        self.model = models.build_backbone(self.net_cfg)
        self.out_channels = [net_config.OUT_CHANNELS[i+1] for i in self.net_cfg['out_indices']]
        self.model.conv1.padding=(0,0) # for Siamese RPN only
        self.model.maxpool.padding=(0,0) # for Siamese RPN only
        if init_weights:
            self.model.init_weights(pretrained = net_config.WEIGHTS_LINK)

    def forward(self,x):
        """
        return: tuple(Tensor)
        """

        return self.model(x)

def build_resnet18(cfg, init_weights = True):
    return Backbone(cfg.MODEL.BACKBONE.RESNET18, init_weights)
def build_resnet50(cfg, init_weights = True):
    return Backbone(cfg.MODEL.BACKBONE.RESNET50, init_weights)

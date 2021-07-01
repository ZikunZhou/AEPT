import os

from yacs.config import CfgNode as CN
#YACS was created as a lightweight library to define and manage system configurations
# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.MODEL = CN()
_C.MODEL.USE_FPN = False
_C.MODEL.DEVICE = "cuda"
_C.MODEL.GPU_NUM = 1


# -----------------------------------------------------------------------------
# Backbone
# -----------------------------------------------------------------------------
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.RESNET50 = CN()
_C.MODEL.BACKBONE.RESNET50.TYPE = 'ResNet'
_C.MODEL.BACKBONE.RESNET50.DEPTH = 50
_C.MODEL.BACKBONE.RESNET50.NUM_STAGES = 3
_C.MODEL.BACKBONE.RESNET50.STRIDES = (1,2,1)
_C.MODEL.BACKBONE.RESNET50.DILATIONS = (1,1,2)
_C.MODEL.BACKBONE.RESNET50.OUT_INDICES = (2,)
_C.MODEL.BACKBONE.RESNET50.FROZEN_STAGES = 3
_C.MODEL.BACKBONE.RESNET50.DCN = CN()
_C.MODEL.BACKBONE.RESNET50.DCN.MODULATED = False
_C.MODEL.BACKBONE.RESNET50.DCN.DEFORMABLE_GROUPS = 1
_C.MODEL.BACKBONE.RESNET50.DCN.FALLBACK_ON_STRIDE = False
_C.MODEL.BACKBONE.RESNET50.STAGE_WITH_DCN = (False, False, False)
_C.MODEL.BACKBONE.RESNET50.WEIGHTS_LINK = 'modelzoo://resnet50'
_C.MODEL.BACKBONE.RESNET50.OUT_CHANNELS = [64, 256, 512, 1024, 2048]

_C.MODEL.BACKBONE.RESNET18 = CN()
_C.MODEL.BACKBONE.RESNET18.TYPE = 'ResNet'
_C.MODEL.BACKBONE.RESNET18.DEPTH = 18
_C.MODEL.BACKBONE.RESNET18.NUM_STAGES = 3
_C.MODEL.BACKBONE.RESNET18.STRIDES = (1,2,1)
_C.MODEL.BACKBONE.RESNET18.DILATIONS = (1,1,2)
_C.MODEL.BACKBONE.RESNET18.OUT_INDICES = (1,2)
_C.MODEL.BACKBONE.RESNET18.FROZEN_STAGES = 3
_C.MODEL.BACKBONE.RESNET18.DCN = CN()
_C.MODEL.BACKBONE.RESNET18.DCN.MODULATED = True
_C.MODEL.BACKBONE.RESNET18.DCN.DEFORMABLE_GROUPS = 1
_C.MODEL.BACKBONE.RESNET18.DCN.FALLBACK_ON_STRIDE = False
_C.MODEL.BACKBONE.RESNET18.STAGE_WITH_DCN = (False, False, False)
_C.MODEL.BACKBONE.RESNET18.WEIGHTS_LINK = 'modelzoo://resnet18'

# ---------------------------------------------------------------------------- #
# FCOS Options
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()
_C.MODEL.FCOS.NUM_CLASSES = 1
if _C.MODEL.USE_FPN:
    _C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
    _C.MODEL.FCOS.LOC_SHIFT = [31, 62, 124, 248, 496]
else:
    _C.MODEL.FCOS.FPN_STRIDES = [8, ]
    _C.MODEL.FCOS.LOC_SHIFT = [31, ]
    _C.MODEL.FCOS.GAMMA = [4,]
_C.MODEL.FCOS.PRIOR_PROB = 0.01
_C.MODEL.FCOS.LOSS_ALPHA = 0.25
_C.MODEL.FCOS.LOSS_GAMMA = 2.0

# the number of convolutions used in the cls and bbox tower
_C.MODEL.FCOS.NUM_CONVS = 4
_C.MODEL.FCOS.NUM_AC = 4
_C.MODEL.FCOS.DILATION = [8, 4, 2, 1]

_C.MODEL.FCOS.GA_SIGMA1 = 0.2
_C.MODEL.FCOS.GA_SIGMA2 = 0.5

# ---------------------------------------------------------------------------- #
# SKConv Options
# ---------------------------------------------------------------------------- #
_C.MODEL.SKC = CN()
_C.MODEL.SKC.BRANCH = 4
_C.MODEL.SKC.GROUP = 8
_C.MODEL.SKC.DOWN_RATIO = 2#need to be >= 1
_C.MODEL.SKC.DILATION = True

# ---------------------------------------------------------------------------- #
# SeLayer Options
# ---------------------------------------------------------------------------- #
_C.MODEL.SE = CN()
_C.MODEL.SE.DOWN_RATIO = 4 #need to be >=1

_C.MODEL.LOSS = CN()
_C.MODEL.LOSS.LAMBDA = 0.6

# ---------------------------------------------------------------------------- #
# Specific siamese options
# ---------------------------------------------------------------------------- #
_C.MODEL.SIAM = CN()
_C.MODEL.SIAM.CROP_SIZE = (4,4)


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TRACKING = CN()
# Number of detections per image
_C.TRACKING.POST_TOPN = 8
_C.TRACKING.MINI_SIZE = 0

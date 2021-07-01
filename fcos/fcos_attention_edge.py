#for fcos
import math
import torch
import torch.nn.functional as F
from torch import nn

from .tracking_edge_attention import make_tracking_postprocessor
from .multi_scale_module import RFAUnit


def init_weights(l):
    if isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear):
        torch.nn.init.normal_(l.weight, std=0.01)
        torch.nn.init.constant_(l.bias, 0)

class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels, out_channels):
        """
        function: FCOSHead
        Arguments:
            in_channels (int): if using cls_neck: number of channels of the input feature , it need equal to the output channels of siamese network
                               if not, not neccessary
            out_channels (int): number of channels　of the hidden layers and output layers in fcos head, if no cls_neck and reg_neck,
                                it need equal to the output channels of siamese network
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES# for single object tracking, num_classes = 1
        self.gamma = cfg.MODEL.FCOS.GAMMA
        #the last convolution layer for output of classification or regression
        #nn.Conv2d(hidden, out_channels, kernel_size=1)
        self.cls_neck = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=True, padding=1),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU(inplace=True))
        self.cls_towers = RFAUnit(cfg, out_channels, out_channels, single_conv=True)

        self.reg_neck = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=True, padding=1),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU(inplace=True))
        self.reg_towers = RFAUnit(cfg, out_channels, out_channels, single_conv=True)

        self.edge_attent = nn.Sequential(
                                    nn.Conv2d(out_channels, out_channels, kernel_size=1),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(out_channels, 4, kernel_size=1))
        self.cls_logits = nn.Conv2d(
            out_channels, num_classes, kernel_size=3, stride=1, padding=1)

        self.reg_pred = nn.Conv2d(
            out_channels, 4, kernel_size=3, stride=1, padding=1)

        self.centerness = nn.Conv2d(
            out_channels, 1, kernel_size=3, stride=1, padding=1)

        # initialization
        for module in [self.cls_neck, self.reg_neck,
                        self.cls_towers, self.reg_towers,
                        self.cls_logits, self.reg_pred,
                        self.centerness, self.edge_attent]:
            print(module)
            module.apply(init_weights)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        #bias_value = -4.59511985013459
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.strides = cfg.MODEL.FCOS.FPN_STRIDES


    def forward(self, x):
        """
        args:
            x (list[Tensor]) - len(list) = num_level, Tensor.shape = [num_srch_frames*batch, channel, height, width]
        returns:
            logits (list[Tensor]) - len(list) = num_level, Tensor:the cls score at each location, shape = [num_srch_frames*batch,class,height,width]
            bbox_reg (list[Tensor]) - len(list) = num_level, Tensor:the regression prediction (l,t,r,b) at each location,
                                      shape = [num_srch_frames*batch,4,height,width]
                                      the predicted bbox by fcos is class-agnostic
            centerness (list[Tensor]) - len(list) = num_level, Tensor:the centerness at each location, shape = [num_srch_frames*batch,class,height,width]
        Note: for single object tracking, num_class = 1(background class is not considered)
        """

        logits = []
        bbox_reg = []
        centerness = []
        edge_attention = []
        for l, feature in enumerate(x):
            cls_neck_feature = self.cls_neck(feature)
            reg_neck_feature = self.reg_neck(feature)
            cls_feature = self.cls_towers(cls_neck_feature)
            reg_feature = self.reg_towers(reg_neck_feature)

            edge_attention.append((self.edge_attent(reg_neck_feature)).sigmoid())
            logits.append(self.cls_logits(cls_feature))
            centerness.append(self.centerness(cls_feature))
            bbox_reg.append(self.gamma[l]*self.strides[l]*torch.exp(
                self.reg_pred(reg_feature)
            ))

        return logits, bbox_reg, centerness, edge_attention


class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels, out_channels):
        """
        Arguments:
            in_channels (int): if using cls_neck: number of channels of the input feature , it need equal to the output channels of siamese network
                               if not, not neccessary
            out_channels (int): number of channels　of the hidden layers and output layers in fcos head, if no cls_neck and reg_neck,
                                it need equal to the output channels of siamese network
        """
        super(FCOSModule, self).__init__()

        self.locations = None
        self.feature_sizes = []
        head = FCOSHead(cfg, in_channels, out_channels)
        tracking_box_selector = make_tracking_postprocessor(cfg)

        self.head = head
        self.tracking_box_selector = tracking_box_selector
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.loc_shift = cfg.MODEL.FCOS.LOC_SHIFT

    def forward(self, image_size, features, targets=None, tracking_state = None, vis_state = None):
        """
        function: perform fcosmodule forward

        Arguments:
            image_size (numpy.ndarray) - the size of the images for which we want to tracking, coordinate definition [height, width]

            features (list[Tensor]) - features computed from the images that are used for computing the
                                      predictions. Each tensor in the list correspond to different feature levels
                                      len(list) = num_level, Tensor.shape = [num_srch_frames, batch, channel, h, w]
            targets (list[BoxList) - ground-truth boxes present in the image (optional),
                                     every BoxList contains the gt-bboxes of an image, for single object tracking, only one gt-bbox,
                                     mode = xyxy, len(list) = num_srch_frames*batch

        Returns:
            boxes (list[BoxList]) - the predicted boxes from the RPN, one BoxList per image.
            losses (dict[Tensor]) - the losses for the model during training. During
                                    testing, it is an empty dict.
        """
        assert features[0].dim() == 5, 'Expect 5  dimensional features'
        num_srch_frames, batch = features[0].shape[0:2]
        features = [f.view(batch * num_srch_frames, f.shape[2], f.shape[3], f.shape[4])for f in features]
        features = [torch.cat(features,dim=1)]

        box_cls, box_regression, centerness, edge_attention = self.head(features)

        if not self.locations or not self.feature_sizes:
            self.locations = self.compute_locations(features)

        return self._forward_test(
            self.locations, box_cls, box_regression,
            centerness, edge_attention, torch.tensor(image_size), self.feature_sizes, tracking_state
        )

    def _forward_test(self, locations, box_cls, box_regression, centerness, edge_attention, image_size, feature_sizes, tracking_state):
        target_pos, target_size, score = self.tracking_box_selector(
            locations, box_cls, box_regression,
            centerness, edge_attention, image_size, feature_sizes, tracking_state
        )

        return target_pos, target_size, score

    def compute_locations(self, features):
        """
        args:
            features (list[Tensor]) - each tensor in this list correspond to a FPN level, len(list) = num_level
        """
        locations = []
        for level, feature in enumerate(features):
            feature_size = torch.tensor(feature.size()[-2:])
            self.feature_sizes.append(feature_size)
            h, w = feature_size# feature的高宽
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device,
                self.loc_shift[level]
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device, loc_shift):
        """
        args:
            h - the height of the siamese feature
            w - the width of the siamese feature
            stride - the stride of the siamese feature
            device - the device of the siamese features
            loc_shift -　we neeed to remove invalid location in the boundary region caused by correlation operation
        returns:
            locations (Tensor) - coordinate defination [x, y], device = feature.device
        Note: fcos header cannot regress the bbox from the anchor in the boundary region. To mitigate this problem, we
              can try to decrease the total_stide to decrease the 'loss_cls_shift'
        """
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + loc_shift
        return locations


def build_fcos(cfg, in_channels, out_channels):
    return FCOSModule(cfg, in_channels, out_channels)

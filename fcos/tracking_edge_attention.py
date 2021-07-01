import torch
import numpy as np

from structures.bounding_box import BoxList
from structures.boxlist_ops import cat_boxlist
from structures.boxlist_ops import boxlist_nms
from structures.boxlist_ops import remove_small_boxes

TO_REMOVE = 0

class TrackingPostProcessor(torch.nn.Module):
    """
    Note:
        1.all postions/locations should be 0-index in calculation! please check it!
        2.for now, we can only test one sequence at a time, because the lengthes of different sequences are different.
    """
    def __init__(
        self,
        cfg,
        post_penal_top_n,
        min_size,
    ):
        """
        Arguments:
            post_penal_top_n (int) - after penalty processing, we select the number of top_n pscores to abtain the final tracking result
            min_size (int) - we need to remove the bboxes whose size is smaller than threshold 'min_size'
        Note: leave the parameter 'min_size' for now
        """
        super(TrackingPostProcessor, self).__init__()
        self.cfg = cfg
        self.post_penal_top_n = post_penal_top_n
        self.min_size = min_size

    def forward_for_single_feature_map(
        self, level, location, box_cls,
        box_regression, centerness, edge_attention, feature_size,
        image_size, tracking_state
        ):
        """
        function: do forward for only one level of FPN features
        args:
            level
            locations (Tensor) - the points on raw_image which are corresponding to the pixels on feature map,
                                 shape = [H×W, 2], coordinate definition [x, y]
            box_cls (Tensor) - shape = [num_srch_frames*batch, num_class, height,width], actually, num_srch_frames*batch=1, num_class=1
            box_regression (Tensor)　-  shape = [num_srch_frames*batch, 4, height,width], coordinate definition: [l,t,r,b]
            centerness (Tensor) - the predicted centerness at each location shape = [num_srch_frames*batch, 1, height,width]
            edge_attention (Tensor) -
            image_size (Tensor) - coordinate definition [height, width]
            tracking_state -
        returns:
            boxlist (BoxList) - penalized tracking result of a single feature map
        Note:
            we need to copy the tensor to cpu
            there is no threshold for cls, we just pick the highest one
            we only tracking one frame of a single sequence for one time, num_srch_frames*batch=1 for tracking
        """
        N, C, H, W = box_cls.shape
        assert N == 1 and C == 1, "Expect only 1 image to track and 1 objec to track"
        #put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(-1).sigmoid().cpu().detach()

        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(-1, 4).cpu().detach()

        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(-1).sigmoid().cpu().detach()

        edge_attention = edge_attention.view(N, 4, H, W).permute(0, 2, 3, 1)
        edge_attention = edge_attention.reshape(-1, 4).cpu().detach()

        location = location.cpu().detach()
        raw_scores = box_cls.clone()# only for debug
        # multiply the classification scores with centerness scores
        box_cls = (box_cls * centerness).to(torch.double)

        #detections.shape = [H×W, 4]
        detections = torch.stack([
            location[:, 0] - box_regression[:, 0],#x_c-l
            location[:, 1] - box_regression[:, 1],#y_c-t
            location[:, 0] + box_regression[:, 2],#x_c+r
            location[:, 1] + box_regression[:, 3],#y_c+b
        ], dim=1).to(torch.double)

        h, w = image_size
        boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
        boxlist.add_field("scores", box_cls)
        boxlist.add_field("edge_attention", edge_attention)
        boxlist.add_field("centerness", centerness)
        boxlist.add_field('box_regression', box_regression)

        boxlist, frame_state = self.cal_penalty_and_roi_info_fsfm(level, tracking_state, boxlist, feature_size)
        target_pos, target_size = self.post_prediction_process(boxlist, location, frame_state, tracking_state, feature_size)

        return target_pos, target_size

    def cal_penalty_and_roi_info_fsfm(self, level, tracking_state, boxlist, feature_size):
        """
        function: calculate penalized scores for single level and
        args:
            tracking_state (dict) - please see method 'forward' in class 'GenTracker'
            boxlist (BoxList) - raw_tracking result of a single feature map, coordinate definition [x1,y1,x2,y2]
                                field:
                                    scores (tensor) - shape [feaure_size*feature_size]
            feature_size (tensor) - the size of the siamese feature map, coordinate defination [height, width]
        returns:
            boxlist (BoxList) - penalized tracking result of a single feature map
                                field:
                                    pscores (Tensor) - shape = [height*width]
                                    delta (Tensor) - shape = [height*width,4]
                                    best_pscore_id - the index of the max-score anchor point
            frame_state (dict) - the state dict of current frame
                                 keys:
                                    corner (tensor) - the corner of the roi in current frame
                                    roi_width (tensor) - the width of the roi in current frame
                                    roi_height (tensor) - the height of the roi in the current frame
        Note:
            the post-processing for different level may need different proposal_selection super parameters
            the arithmetical operations between different numpy.dtype are compatable
            the arithmetical operations between different torch.dtype are not compatable
        """
        target_size = torch.tensor(tracking_state['target_size'], dtype = torch.double)
        scale_template = torch.tensor(tracking_state['scale_template'], dtype = torch.double)
        window = torch.tensor(tracking_state['window'])
        tracking_config = tracking_state['tracking_config']

        bbox = boxlist.bbox
        scores = boxlist.get_field("scores")
        delta = torch.stack([(bbox[:,0] + bbox[:,2] - (tracking_config.instance_size-TO_REMOVE))/2.,
                          (bbox[:,1] + bbox[:,3] - (tracking_config.instance_size-TO_REMOVE))/2.,
                          bbox[:,2] - bbox[:,0] + TO_REMOVE,
                          bbox[:,3] - bbox[:,1] + TO_REMOVE,
                        ],dim = 1).to(torch.double)
        target_size = target_size * scale_template

        scale_c = self.change(self.cal_size(delta[:,2], delta[:,3]) / self.cal_size(target_size[0], target_size[1]))
        ratio_c = self.change((target_size[0] / target_size[1]) / (delta[:,2] / delta[:,3]))
        penalty = torch.exp(-(scale_c * ratio_c -1.) * tracking_config.penalty_k).to(torch.double)

        pscores = penalty * scores
        pscores = pscores * (1 - tracking_config.window_influence) + window * tracking_config.window_influence
        best_pscore_id = torch.argmax(pscores)
        corner, roi_width, roi_height = self.cal_roi_info(target_size, best_pscore_id, feature_size, self.cfg.MODEL.FCOS.FPN_STRIDES[level])
        frame_state = {}
        boxlist.add_field('pscores', pscores)
        boxlist.add_field('best_pscore_id', best_pscore_id)
        boxlist.add_field('delta', delta)
        boxlist.add_field('penalty', penalty)
        frame_state['corner'] = corner
        frame_state['roi_width'] = roi_width
        frame_state['roi_height'] = roi_height
        return boxlist, frame_state

    def post_prediction_process(self, boxlist, location, frame_state, tracking_state, feature_size):
        """
        function: for the prediction of each edge, we average the predictions of all anchors in the roi
                  with edge_attention and pscores as weights
        args:
            boxlist (BoxList) - penalized tracking result of a single feature map
                            field:
                                edge_attention (Tensor) -
                                box_regression (Tensor) -
                                pscores (Tensor) - shape = [height*width]
                                delta (Tensor) - shape = [height*width,4]
                                best_pscore_id - the index of the max-score anchor point
        returns:
            target_pos (numpy.ndarray)
            target_size (numpy.ndarray)
        """

        box_regression = boxlist.get_field('box_regression')
        edge_attention = boxlist.get_field('edge_attention')
        prediction = {}
        prediction['left'] = (location[:, 0] - box_regression[:, 0]).to(torch.double)#x_c-l
        prediction['top'] = (location[:, 1] - box_regression[:, 1]).to(torch.double)#y_c-t
        prediction['right'] = (location[:, 0] + box_regression[:, 2]).to(torch.double)#x_c+r
        prediction['bottom'] = (location[:, 1] + box_regression[:, 3]).to(torch.double)#y_c+b

        attention_weight = {}
        attention_weight['left'] = edge_attention[:, 0].to(torch.double)
        attention_weight['top'] = edge_attention[:, 1].to(torch.double)
        attention_weight['right'] = edge_attention[:, 2].to(torch.double)
        attention_weight['bottom'] = edge_attention[:, 3].to(torch.double)

        corner, roi_width, roi_height = frame_state['corner'], frame_state['roi_width'], frame_state['roi_height']
        pscores = boxlist.get_field('pscores')
        best_pscore_id = boxlist.get_field('best_pscore_id')
        penalty = boxlist.get_field('penalty')
        scores = boxlist.get_field('scores')
        indices_in_roi = self.cal_location_index_in_roi(corner, feature_size)
        detection = {}
        for edge_name in prediction.keys():
            prediction_edge = prediction[edge_name][indices_in_roi]
            weight_edge = attention_weight[edge_name][indices_in_roi]
            pscores_in_roi = pscores[indices_in_roi]
            if indices_in_roi.numel() > self.post_penal_top_n:
                _, topk_index = torch.topk(weight_edge * pscores_in_roi, self.post_penal_top_n)
                post_weight_edge = weight_edge[topk_index]
                post_pscores = pscores_in_roi[topk_index]
            else:
                topk_index = torch.arange(len(indices_in_roi))
                post_weight_edge = weight_edge
                post_pscores = pscores_in_roi
            detection[edge_name] = torch.sum(prediction_edge[topk_index]*post_pscores*post_weight_edge) / torch.sum(post_pscores*post_weight_edge)
        detection = torch.tensor([detection[key] for key in detection.keys()]).to(torch.double)
        target_pos = torch.tensor(tracking_state['target_pos'], dtype = torch.double)
        target_size = torch.tensor(tracking_state['target_size'], dtype = torch.double)
        scale_template = torch.tensor(tracking_state['scale_template'], dtype = torch.double)
        tracking_config = tracking_state['tracking_config']

        delta = torch.stack([(detection[0] + detection[2] - (tracking_config.instance_size-TO_REMOVE))/2.,
                          (detection[1] + detection[3] - (tracking_config.instance_size-TO_REMOVE))/2.,
                          detection[2] - detection[0] + TO_REMOVE,
                          detection[3] - detection[1] + TO_REMOVE,
                        ]).to(torch.double)

        target = delta / scale_template

        lr = penalty[best_pscore_id] * scores[best_pscore_id] * tracking_config.lr

        res_x = target[0] + target_pos[0]
        res_y = target[1] + target_pos[1]
        res_w = target_size[0] * (1 - lr) + target[2] * lr
        res_h = target_size[1] * (1 - lr) + target[3] * lr

        target_pos = np.array([res_x, res_y])
        target_size = np.array([res_w, res_h])
        return target_pos, target_size


    def cal_location_index_in_roi(self, corner, feature_size):
        """
        function: we consider the locations in such a region: its size is the same as the object on last frame and it is centerded
                  around the location with best_pscore_id;
                  this function will calculate the indices of the anchors in this region
        args:
            corner (tensor) - [tl_x, tl_y, br_x, br_y], 0-index
            feature_size (Tensor) - [height, width]
        returns:
            index (Tensor) - index of in-box anchor in flatterm mode
        """
        f_h, f_w = feature_size
        tl_x, tl_y, br_x, br_y = corner
        #if tl_x < 0:
        #    tl_x = 0
        #if tl_y < 0:
        #    tl_y = 0
        #if br_x >= f_w:
        #    br_x = f_w-1
        #if br_y >= f_h:
        #    br_y = f_h-1
        tf_h = br_y - tl_y + 1
        tf_w = br_x - tl_x + 1
        index = []
        for i in range(tf_h):
            index.append(torch.arange((tl_y+i)*f_w+tl_x, (tl_y+i)*f_w+br_x+1))
        index = torch.cat(index, dim = 0)
        return index


    def cal_roi_info(self, target_size, best_pscore_id, feature_size, stride):
        """
        function: we consider the locations in such a region: its size is the same as the object on last frame and it is centerded
                  around the location with best_pscore_id;
                  this function will calculate the corner coordinate of this region
        args:
            target_size (Tensor) - [width, height]
            feature_size (int) - feature map need to be square
        returns:
            corner (Tensor) - 0-index
            tf_w (Tensor)
            tf_h (Tensor)
        """
        f_h, f_w = feature_size
        y, x = ((best_pscore_id+1)/f_w), (best_pscore_id+1).fmod(f_w)-1#0-index

        tf_w, tf_h = torch.floor(target_size/stride).to(torch.int)
        if tf_w == 0:
            tf_w += 1
        if tf_h == 0:
            tf_h += 1
        if tf_w.fmod(2) == 0:
            tf_w = tf_w - 1
        if tf_h.fmod(2) == 0:
            tf_h = tf_h - 1

        tl_x, tl_y, br_x, br_y = x-(tf_w-1)/2, y-(tf_h-1)/2, x+(tf_w-1)/2, y+(tf_h-1)/2
        if tl_x < 0:
            tl_x = 0
        if tl_y < 0:
            tl_y = 0
        if br_x >= f_w:
            br_x = f_w-1
        if br_y >= f_h:
            br_y = f_h-1
        tf_w = br_x - tl_x + 1
        tf_h = br_y - tl_y + 1
        return torch.tensor([tl_x, tl_y, br_x, br_y]), tf_w, tf_h


    def change(self, r):
        return torch.max(r, 1./r)

    def cal_size(self, width, height):
        if isinstance(height, np.float64):
            height = torch.tensor(height)
            width = torch.tensor(width)
        pad = (height + width)/2.
        size_2 = (height + pad) * (width + pad)
        return torch.sqrt(size_2)

    def forward(self, locations, box_cls, box_regression, centerness, edge_attention, image_size, feature_sizes, tracking_state):
        """
        function: perform tracking post processing. We choose the anchor points in the bbox which is the same size of the last frame but center on
                  the max-score anchor in current frame. The weights of each anchor points depend on the pscores and edge_attention
        args:
            locations (list[Tensor]) - coordinate definition [x,y], len(list) = num_level,
                                       Tensor:the points on raw_image which are corresponding to the pixels on feature map,
                                       shape = [num_level_point,2]
            box_cls (list[Tensor]) - len(list) = num_level, Tensor:the cls score at each location, shape = [num_srch_frames*batch,class,height,width]
            box_regression (list[Tensor]) - len(list) = num_level, Tensor:Tensor: the predicted (l,t,r,b) at each location,
                                            shape = [num_srch_frames*batch,4,height,width]
            edge_attention (list[Tensor]) - len(list) = num_level, Tensor: we set a weight to each prediction edge of each anchor point, and we call it
                                            edge_attention, its value is between 0 and 1
            centerness (list[Tensor]) - len(list) = num_level, Tensor:the predicted centerness at each location,
                                        shape = [num_srch_frames*batch,class,height,width]
            image_size: Tensor - coordinate definition [height width], image size of the whole　tracking image
            feature_size: Tensor - coordinate format:[height, width]
            tracking_state (dict) - please see method 'forward' in class 'GenTracker'
                                    keys: depending on class 'TASiam_Tracker'
        returns:

        """
        sampled_boxes = []
        for level, (l, o, b, c, e, f_s) in enumerate(zip(locations, box_cls, box_regression, centerness, edge_attention, feature_sizes)):
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    level, l, o, b, c, e, f_s, image_size, tracking_state
                )
            )
        target_pos, target_size = sampled_boxes[0]

        return target_pos, target_size, None

    def generate_gauss_label(self, size, sigma, center = (0, 0), end_pad=(0, 0)):
        """
        function: generate gauss label for L2 loss
        """

        shift_x = torch.arange(-(size[1] - 1) / 2, (size[1] + 1) / 2 + end_pad[1])
        shift_y = torch.arange(-(size[0] - 1) / 2, (size[0] + 1) / 2 + end_pad[0])

        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        alpha = 0.2
        gauss_label = torch.exp(-1*alpha*(
                            (shift_y-center[0])**2/(sigma[0]**2) +
                            (shift_x-center[1])**2/(sigma[1]**2)
                            ))
        return gauss_label

def make_tracking_postprocessor(config):

    tracking_box_selector = TrackingPostProcessor(
        config,
        post_penal_top_n=config.TRACKING.POST_TOPN,
        min_size=config.TRACKING.MINI_SIZE
    )

    return tracking_box_selector

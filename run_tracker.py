
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import os

import sys
from torchvision import transforms
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from tracking_utils.tracking_utils import get_subwindow, cxy_wh_2_rect, get_anno
from tracking_utils.image_loader import default_image_loader
from admin import loading
from pyvotkit.region import vot_overlap, vot_float2str

TO_REMOVE = 0

env_path = os.path.join(os.path.dirname(__file__), '..')

if env_path not in sys.path:
    sys.path.append(env_path)

debug_flag = False

fig,ax = plt.subplots(1,4)

class TrackerConfig(object):
    def __init__(self, penalty_k = 0.04, window_influence = 0.42, lr = 0.25, score_size = None):
        # These are the default hyper-params for DaSiamRPN 0.3827
        self.windowing = 'cosine'  # to penalize large displacements [cosine/uniform]
        # Params from the network architecture, have to be consistent with the training
        self.exemplar_size = 127# input z size
        if score_size is None:
            self.instance_size = 255# input x size (search region)
            self.score_size = 25
        else:
            self.instance_size = 55 + 8*score_size
            self.score_size = score_size
        self.total_stride = 8
        self.context_amount = 0.5  # context amount for the exemplar
        self.penalty_k = penalty_k
        self.window_influence = window_influence
        self.lr = lr
    def __str__(self):
        return "penal%.2f_win%.2f_lr%.2f"%(self.penalty_k, self.window_influence, self.lr)

    def __repr__(self):
        return "penal%.2f_win%.2f_lr%.2f"%(self.penalty_k, self.window_influence, self.lr)

    def print_cfg(self):
        print('windowing:         ',self.windowing)
        print('exemplar_size:     ',self.exemplar_size)
        print('instance_size:     ',self.instance_size)
        print('total_stride:      ',self.total_stride)
        print('score_size:        ',self.score_size)
        print('context_amount:    ',self.context_amount)
        print('penalty_k:         ',self.penalty_k)
        print('window_influence:  ',self.window_influence)
        print('lr:                ',self.lr)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def unnormal(tensor):
    return (((tensor.cpu().squeeze(0).numpy().transpose((1,2,0))+np.array([0.229,0.224,0.225]))*np.array([0.485,0.456,0.406]))*255+20).clip(0,255).astype(np.uint8)

class TASiam_Tracker(object):
    """Note: all postions/locations should be 0-index in calculation! please check it!"""
    def __init__(self, tracking_config, cfg, model = None, device = 'cpu', name = 'TASiam', checkpoint = None, display = False):
        """
        args:
            tracking_config (TrackerConfig) - tracker super parameters
            model - tracking model
            checkpoint (str or int) - str means absolute path, int means checkpoint number
            display (bool) - whether to display the tracking bbox in sequence
        Note: all coordinate should be 0-index
        """
        super(TASiam_Tracker, self).__init__()
        self.name = name
        self.device = device
        self.display = display
        self.model = model
        self.config = cfg
        #print(self.model)
        self.load_checkpoint(self.model, checkpoint)
        self.model.to(self.device).eval()
        self.t_cfg = tracking_config
        self.tracking_state = {'tracking_config': tracking_config}
        self.results = []
        #print(self.t_cfg)
        self.toc = 0

    def load_checkpoint(self, model, checkpoint):
        """
        function: load checkpoint for tracking model
        """
        root = '/home/zikun/work/tracking/gan_tracker/train_gen_tracker/checkpoints/gan_tracker/gen_tracker/'
        if isinstance(checkpoint, int):
            #TODO
            assert root is not None, "please set the root path of checkpoints"
            checkpoint_list = sorted(glob.glob(root + '*.pth.tar'))
            checkpoint_path = checkpoint_list[int]
        elif isinstance(checkpoint, str):
            checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError
        print('loading checkpoint form ' + checkpoint_path)
        checkpoint_dict = loading.torch_load_legacy(checkpoint_path)
        assert checkpoint_dict['net_type'] == "Generator", 'Network is not of correct type.'
        #model.load_state_dict(checkpoint_dict['net'], strict=True)
        model.load_state_dict(checkpoint_dict['net'])

    def tasiam_init(self, image, target_pos, target_size, gt):
        """
        function: initialize tasiamese tracker
        args:
            image (numpy.array or path) -
            target_pos (numpy.array) - coordinate definition [x_c, y_c], should be 0-index
            target_size (numpy.array) - coordinate definition　[width, hight]
            gt (numpy.array) - coordinate definition: if len(gt) = 8, [tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, l_y];
                               if len(gt)=4, [x1,y1,w,h].
                               This parameter is not only for visaulization, it is involved in the exemplar image crop
        """
        if isinstance(image, str) and os.path.exists(image):
            image = default_image_loader(image)#<class 'numpy.ndarray'> [height, width, channel]
        else:
            assert isinstance(image, np.ndarray) and image.shape[-1] == 3, "The input image need to be directory or numpy.ndarray"

        self.tracking_state['image_h'] = image.shape[0]
        self.tracking_state['image_w'] = image.shape[1]
        avg_chans = np.mean(image, axis=(0, 1))
        wc_z = target_size[0] + self.t_cfg.context_amount * sum(target_size)
        hc_z = target_size[1] + self.t_cfg.context_amount * sum(target_size)
        patch_size = round(np.sqrt(wc_z * hc_z))

        ref_image = get_subwindow(image, target_pos, patch_size, self.t_cfg.exemplar_size, avg_chans, visualize = False)
        #only for debug
        self.ref_anno_gt = get_anno(target_pos, gt, patch_size, self.t_cfg.exemplar_size)

        ref_image = transform(ref_image).unsqueeze(0).unsqueeze(0).to(self.device)
        #print('ref_image requires_grad in run_tracker:',ref_image.requires_grad)
        self.model.temple(ref_image)
        if self.t_cfg.windowing == 'cosine':
            window = np.outer(np.hanning(self.t_cfg.score_size), np.hanning(self.t_cfg.score_size))
        elif self.t_cfg.windowing == 'uniform':
            window = np.ones((self.t_cfg.score_size, self.t_cfg.score_size))

        window = window.flatten()


        self.tracking_state['avg_chans'] = avg_chans
        self.tracking_state['window'] = window
        self.tracking_state['target_pos'] = target_pos
        self.tracking_state['target_size'] = target_size

        if self.display and gt is not None:
            self.visaulization(0, gt, image, gt-np.array([1,1,0,0]))
        self.results.append(gt)
        return self.tracking_state


    def tasiam_tracking(self, image, gt, frame = 1):
        """
        args:
            image: frame to track
            gt: groundtruth of this frame, only for visualization
        returns:
            self.tracking_state
        """
        if isinstance(image, str) and os.path.exists(image):
            image = default_image_loader(image)#<class 'numpy.ndarray'> [height, width, channel]
        else:
            assert isinstance(image, np.ndarray) and image.shape[-1] == 3, "The input image need to be directory or numpy.ndarray"
        tic = cv2.getTickCount()
        avg_chans = self.tracking_state['avg_chans']
        target_pos = self.tracking_state['target_pos']
        target_size = self.tracking_state['target_size']

        wc_z = target_size[0] + self.t_cfg.context_amount * sum(target_size)
        hc_z = target_size[1] + self.t_cfg.context_amount * sum(target_size)
        tem_patch_size = np.sqrt(wc_z * hc_z)

        scale_template = self.t_cfg.exemplar_size / tem_patch_size
        d_search = (self.t_cfg.instance_size - self.t_cfg.exemplar_size) / 2
        pad = d_search / scale_template
        srch_patch_size = tem_patch_size + 2 * pad
        srch_image = get_subwindow(image, target_pos, round(srch_patch_size), self.t_cfg.instance_size, avg_chans)

        srch_anno_gt = get_anno(target_pos, gt, round(srch_patch_size), self.t_cfg.instance_size)

        srch_image = transform(srch_image).unsqueeze(0).unsqueeze(0).to(self.device)
        self.tracking_state['scale_template'] = scale_template
        target_pos, target_size, score = self.model(srch_image, image.shape[0:2], self.tracking_state)

        target_pos[0] = max(0, min(self.tracking_state['image_w'], target_pos[0]))
        target_pos[1] = max(0, min(self.tracking_state['image_h'], target_pos[1]))
        target_size[0] = max(10, min(self.tracking_state['image_w'], target_size[0]))
        target_size[1] = max(10, min(self.tracking_state['image_h'], target_size[1]))
        self.tracking_state['target_pos'] = target_pos
        self.tracking_state['target_size'] = target_size
        self.tracking_state['score'] = score
        self.toc += cv2.getTickCount() - tic

        location = cxy_wh_2_rect(target_pos, target_size)
        self.results.append(location)
        if self.display and gt is not None:

            self.visaulization(frame, gt, image, location)
            if debug_flag:
                self.debug(self.ref_anno_gt, srch_anno_gt, score)
        return self.tracking_state

    def cal_time(self):
        toc = self.toc / cv2.getTickFrequency()
        return toc

    def visaulization(self, frame, gt, image, location, lost_times = 0):
        """
        args:
            gt (numpy.ndarray) - coordinate definition: if len(gt) = 8, [tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, l_y];
                                        if len(gt)=4, [x1,y1,w,h]
            image (numpy.ndarray) -
            location (numpy.ndarray) - coordinate definition: if len(location) = 8, [tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, l_y];
                                            if len(location)=4, [x1,y1,w,h]
            lost_times (int) - For VOT benchmark, we should care about lost times
        """

        if gt.dtype is not np.int:
            gt = np.round(gt).astype(np.int)

        gt = gt - np.array([1,1,0,0])
        im_show = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if frame == 0:
            cv2.destroyAllWindows()
        if len(gt) == 8:
            cv2.polylines(im_show, [np.array(gt, np.int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
        else:
            cv2.rectangle(im_show, (gt[0], gt[1]), (gt[0] + gt[2], gt[1] + gt[3]), (0, 255, 0), 3)
        if len(location) == 8:
            location_int = np.int0(location)
            cv2.polylines(im_show, [location_int.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
        else:
            location = [int(l) for l in location]

            cv2.rectangle(im_show, (location[0], location[1]),
                          (location[0] + location[2]-TO_REMOVE, location[1] + location[3]-TO_REMOVE), (0, 255, 255), 3)

        cv2.putText(im_show, str(frame), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(im_show, str(lost_times), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        window = cv2.namedWindow("camera", cv2.WINDOW_NORMAL);
        cv2.imshow("camera", im_show)
        cv2.waitKey(1)

    def debug(self, ref_anno_gt, srch_anno_gt, score):
        """
        Note:
        Different versions of fcos, Siamese code corresponding debug code interface is not unified, TODO
        args:
            ref_anno_gt (numpy.array) - coordinate definition [y1, x1, y2, x2]
            srch_anno_gt (numpy.array) - coordinate definition [y1, x1, y2, x2]
            score (list)
        """
        ref_features, non_crop_ref_features, srch_features, siam_features = self.model.debug()
        ref_feature_heatmap = self.visaulize_feat(ref_features[0])
        srch_feature_heatmap = self.visaulize_feat(srch_features[0])
        siam_feature_heatmap = self.visaulize_feat(siam_features[0])
        non_crop_ref_feature_heatmap = self.visaulize_feat(non_crop_ref_features[0])

        locations = [location.clone().cpu().numpy() for location in self.model.fcos.locations]
        location = locations[0]

        boxlists, best_pscore_id = score[1:]
        pscores = boxlists.get_field("pscores").reshape(25, 25)
        scores = boxlists.get_field("scores")
        penalty = boxlists.get_field("penalty")
        delta = boxlists.get_field("delta")[best_pscore_id,:]
        raw_scores = boxlists.get_field("raw_scores").reshape(25, 25)
        centerness = boxlists.get_field("centerness")

        ax[0].clear()
        ax[0].set_title('heatmap-siam-features')
        ax[0].imshow(siam_feature_heatmap)

        ax[1].clear()
        ax[1].set_title("raw_scores")
        ax[1].imshow(raw_scores)

        ax[2].clear()
        ax[2].set_title('pscores')
        ax[2].imshow(pscores)

        ax[3].clear()
        print(delta)
        ax[3].set_title('srch_image')
        ax[3].imshow(unnormal(self.model.srch_images.squeeze(0)))
        ax[3].add_patch(Rectangle((srch_anno_gt[1],srch_anno_gt[0]),srch_anno_gt[3]-srch_anno_gt[1],srch_anno_gt[2]-srch_anno_gt[0],fill=False,color='g'))
        ax[3].add_patch(Rectangle((delta[0]+128-delta[2]/2, delta[1]+128-delta[3]/2), delta[2],delta[3],fill=False,color='r'))
        plt.pause(1)

    def visaulize_feat(self, feature):
        """
        """
        while True:
            if feature.shape[0] == 1:
                feature = feature.squeeze()
            else:
                break
        heatmap = torch.sum(feature, dim = 0)
        max_value = torch.max(heatmap)
        min_value = torch.min(heatmap)
        heatmap = (heatmap-min_value)/(max_value-min_value)*255
        return heatmap.numpy().astype(np.uint8)

    def visualize_channels(self, features):
        grid = make_grid(features, nrow=8, padding=2, normalize=True, range=None, scale_each=False, pad_value=0)
        ndarr = grid.mul(255).clamp(0, 255).byte().cpu().permute(1, 2, 0).numpy()
        b, g, r = cv2.split(ndarr)
        ndarr = cv2.merge([r, g, b])
        ndarr = ndarr.copy()

    def saving_result(self, result_path, video_name, results = None, zero_index = True):
        """
        args:
            result_path - the root path to restore the tracking results
            video_name
            results - if vot, results is the tacking result of the given video
                      if otb, None
            zero_index -
        """
        if results:#vot
            video_path = os.path.join(result_path, 'baseline', video_name)
            if not os.path.isdir(video_path): os.makedirs(video_path)
            result_path = os.path.join(video_path, '{:s}_001.txt'.format(video_name))
            with open(result_path, "w") as fin:
                for x in results:
                    fin.write("{:d}\n".format(x)) if isinstance(x, int) else \
                            fin.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')
        else:#otb
            if zero_index:
                results = self.results
            else:
                #convert to 1-index
                results = [result + np.array([1,1,0,0]) for result in self.results]
            with open(result_path + video_name +'.txt','w') as f:
                for bbox in results:
                    #bbox coordinate [x1, y1, w, h]
                    newline = str(bbox[0])+','+str(bbox[1])+','+str(bbox[2])+','+str(bbox[3]) + '\n'
                    f.write(newline)

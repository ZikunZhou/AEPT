import sys
sys.path.append('/path/to/pysot-toolkit')
from pysot.datasets import DatasetFactory
from pysot.utils.region import vot_overlap
import numpy as np
import os
import torch
from glob import glob
import cv2
from collections import OrderedDict
import logging

from config import cfg
from run_tracker import TASiam_Tracker, TrackerConfig
from fcos.generator_attention_edge import GenTracker
from tracking_utils.tracking_utils import cxy_wh_2_rect
from tracking_utils.image_loader import default_image_loader

from tqdm import tqdm
from multiprocessing import Pool
from pysot.datasets import OTBDataset, UAVDataset, LaSOTDataset, VOTDataset, NFSDataset, VOTLTDataset
from pysot.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, EAOBenchmark, F1Benchmark
from pysot.visualization import draw_success_precision, draw_eao, draw_f1
from pyvotkit.region import vot_overlap,vot_float2str


TO_REMOVE = 0

def test_vot(vot_root, checkpoint_path, result_root_path, p, device = 'cpu', tracker_base_name = 'AEPT'):
    dataset = load_dataset(base_path = vot_root)
    result_path = os.path.join(result_root_path, tracker_base_name+'/')
    tracker = tracker_base_name
    test_vot_once(checkpoint_path, result_path, p, dataset = dataset, device = device, display = False)
    return trackers

def test_vot_once(checkpoint_path, result_path, p = None, dataset = None, device = 'cpu', display = False):
    if dataset is None:
        dataset = load_dataset(base_path = '/path/to/VOT2016')
    frame_counter = 0
    model = GenTracker(cfg)

    if isinstance(p, dict):
        t_cfg = TrackerConfig(**p)
    elif isinstance(p, list) or p.shape == (3,):#numpy.ndarray
        t_cfg = TrackerConfig(*p)
    else:
        t_cfg = TrackerConfig()
    print(t_cfg)

    if not os.path.exists(result_path):
        os.mkdir(result_path)
    for video_name in dataset.keys():
        print('video name:',video_name)
        regions, lost_times, fps = track_vot(dataset[video_name],t_cfg, model, checkpoint_path, device, display,result_path)

def track_vot(video, t_cfg, model, checkpoint_path, device, display,result_path):
    regions = []# result and states[1 init / 2 lost / 0 skip]
    image_files, gt = video['image_files'], video['gt']
    start_frame, end_frame, lost_times, toc = 0, len(image_files), 0, 0
    tracker = TASiam_Tracker(t_cfg, cfg, model = model, checkpoint = checkpoint_path, device = device, display = display)
    for f, image_path in enumerate(image_files):
        image = default_image_loader(image_path)
        tic = cv2.getTickCount()#for calculate fps

        if f == start_frame:
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])
            target_pos = np.array([cx, cy])
            target_size = np.array([w, h])
            target_location = cxy_wh_2_rect(target_pos, target_size)

            tracker.tasiam_init(image, target_pos, target_size, target_location)
            regions.append(1)
        elif f > start_frame:# tracking
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])
            target_location = cxy_wh_2_rect([cx, cy], [w, h])# for visualization
            state = tracker.tasiam_tracking(image, target_location ,f)
            location = cxy_wh_2_rect(state['target_pos'], state['target_size'])

            gt_polygon = ((gt[f][0], gt[f][1]), (gt[f][2], gt[f][3]),
                          (gt[f][4], gt[f][5]), (gt[f][6], gt[f][7]))
            pred_polygon = ((location[0], location[1]),
                            (location[0] + location[2], location[1]),
                            (location[0] + location[2], location[1] + location[3]),
                            (location[0], location[1] + location[3]))

            b_overlap = vot_overlap(gt_polygon, pred_polygon, (image.shape[1], image.shape[0]))

            if b_overlap:
                regions.append(location)
            else:
                regions.append(2)
                lost_times += 1
                start_frame = f + 5# skip 5 frames
        else:# skip
            regions.append(0)
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    logging.info('(Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                                                video['name'], toc, f / toc, lost_times))
    tracker.saving_result(result_path, video['name'], results = regions, zero_index = False)
    return regions, lost_times, f / toc

def get_axis_aligned_bbox(region):
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2

    return cx, cy, w, h

def load_dataset(base_path = None):
    info = OrderedDict()
    print(base_path)
    if not os.path.exists(base_path):
        logging.error("Please download test dataset!!!")
        exit()
    list_path = os.path.join(base_path, 'list.txt')
    with open(list_path) as f:
        videos = [v.strip() for v in f.readlines()]
    for video in videos:
        video_path = os.path.join(base_path, video)
        image_path = os.path.join(video_path, '*.jpg')
        image_files = sorted(glob(image_path))
        if len(image_files) == 0:  # VOT2018
            image_path = os.path.join(video_path, 'color', '*.jpg')
            image_files = sorted(glob(image_path))
        gt_path = os.path.join(video_path, 'groundtruth.txt')
        gt = np.loadtxt(gt_path, delimiter=',').astype(np.float64)
        if gt.shape[1] == 4:
            gt = np.column_stack((gt[:, 0], gt[:, 1], gt[:, 0], gt[:, 1] + gt[:, 3]-1,
                                  gt[:, 0] + gt[:, 2]-1, gt[:, 1] + gt[:, 3]-1, gt[:, 0] + gt[:, 2]-1, gt[:, 1]))
        info[video] = {'image_files': image_files, 'gt': gt, 'name': video}
    return info

if __name__ == '__main__':
    vot_root = '/path/to/VOT2016'
    checkpoint_path = "/path/to/model/"
    result_root_path = '/root/path/for/saving/results/'
    #hyper-parameters
    p = {'penalty_k': 0.465, 'window_influence': 0.390, 'lr': 0.465}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trackers = test_vot(
                vot_root,
                checkpoint_path,
                result_root_path,
                p,
                device = device,
                tracker_base_name = 'Tracker_name',
            )

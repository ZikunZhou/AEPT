import torch
import numpy as np
from scipy import signal
import cv2
TO_REMOVE = 0

def get_subwindow(image, position, window_size, out_size, avg_chans, visualize = False):

    """
    function: crop the image according to the given location (position, window_size), then resize the image_patch
              to 'input_sz' and transform to tensor
    args:
        image (numpy.ndarray) - shapr = [height, width, chennel]
        position (numpy.ndarray) - subwindow position, coordinate definition [x_c, y_c] 0-index
        window_size (numpy.float64) - subwindow size: square
        out_size (int) - the required resized size, coordinate definition [width,height],
                         we require the input of network to be square, so the coordinate definition is not important
        avg_chans (numpy.ndarray) - the padding value, shape = (3,)
        visualize - whether to visualize the subwindow
    """
    image_size = image.shape[0:2]

    #center = (window_size + 1) / 2
    center = (window_size - TO_REMOVE) / 2
    context_xmin = np.round(position[0] - center)
    context_xmax = context_xmin + window_size - 1
    context_ymin = np.round(position[1] - center)
    context_ymax = context_ymin + window_size - 1

    left_pad = np.max([0., -context_xmin]).astype(int)
    top_pad = np.max([0., -context_ymin]).astype(int)
    right_pad = np.max([0., context_xmax - image_size[1] + 1]).astype(int)
    bottom_pad = np.max([0., context_ymax - image_size[0] + 1]).astype(int)

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    # zzp: a more easy speed version
    height, width, channel = image.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        temp_image = np.zeros((height + top_pad + bottom_pad, width + left_pad + right_pad, channel), np.uint8)  # 0 is better than 1 initialization
        temp_image[top_pad:top_pad + height, left_pad:left_pad + width, :] = image
        if top_pad:
            temp_image[0:top_pad, left_pad:left_pad + width, :] = avg_chans
        if bottom_pad:
            temp_image[height + top_pad:, left_pad:left_pad + width, :] = avg_chans
        if left_pad:
            temp_image[:, 0:left_pad, :] = avg_chans
        if right_pad:
            temp_image[:, width + left_pad:, :] = avg_chans
        im_patch_original = temp_image[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = image[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    if not window_size == out_size:
        image = cv2.resize(im_patch_original, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    else:
        image = im_patch_original
    if visualize:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        print('show_image')
        cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('input_image', image)
        cv2.waitKey(1000)#0或None表示一直显示, 其他数值表示毫秒
        cv2.destroyAllWindows()
    return image

def get_anno(target_pos, gt, window_size, out_size):
    """
    args:
        target_pos (numpy.array) - coordinate definition [x_c,y_c]
        gt (numpty.array) - coordinate definition [x1, y1, w ,h]
        window_size (int) -
        out_size (int) -
    returns:
        bbox (numpy.array) - coordinate definition [y1,x1,y2,x2]
    """
    gt_pos =gt[0:2] + (gt[2:4]-TO_REMOVE)/2
    shift_x, shift_y = gt_pos - target_pos
    #print('shift_x, shift_y:',shift_x, shift_y)
    target_size = gt[-2:] * out_size / window_size
    h, w = target_size[1], target_size[0]
    bbox = [
        (out_size-TO_REMOVE)/2+shift_y-(h-TO_REMOVE)/2,
        (out_size-TO_REMOVE)/2+shift_x-(w-TO_REMOVE)/2,
        (out_size-TO_REMOVE)/2+shift_y+(h-TO_REMOVE)/2,
        (out_size-TO_REMOVE)/2+shift_x+(w-TO_REMOVE)/2,
        ]
    return np.round(np.array(bbox))

def get_anno_for_vis(target_pos, gt, window_size, out_size):
    """
    args:
        target_pos (numpy.array) - coordinate definition [x_c,y_c]
        gt (numpty.array) - coordinate definition [x1, y1, w ,h]
        window_size (int) -
        out_size (int) -
    returns:
        bbox (numpy.array) - coordinate definition [y1,x1,y2,x2]
    """
    gt_pos =gt[0:2] + (gt[2:4]-TO_REMOVE)/2
    shift_x, shift_y = (gt_pos - target_pos) * out_size / window_size
    print('shift_x, shift_y:',shift_x, shift_y)
    target_size = gt[-2:] * out_size / window_size
    h, w = target_size[1], target_size[0]
    bbox = [
        (out_size-TO_REMOVE)/2+shift_y-(h-TO_REMOVE)/2,
        (out_size-TO_REMOVE)/2+shift_x-(w-TO_REMOVE)/2,
        (out_size-TO_REMOVE)/2+shift_y+(h-TO_REMOVE)/2,
        (out_size-TO_REMOVE)/2+shift_x+(w-TO_REMOVE)/2,
        ]
    return np.round(np.array(bbox))

def cxy_wh_2_rect(pos, sz):
    """
    args:
        pos (numpy.ndarray) - [x_c,y_c]
        sz (numpy.ndarray) - [w,h]
    return (numpy.ndarray): [x1, y1, w, h]
    """
    return np.array([pos[0]-(sz[0]-TO_REMOVE)/2, pos[1]-(sz[1]-TO_REMOVE)/2, sz[0], sz[1]])  # 0-index



def rect_2_cxy_wh(rect):
    """
    args:
        rect (numpy.ndarray) - [x1, y1, w, h]
    return:
        (numpy.ndarray),(numpy.ndarray) - [x_c, y_c],[w, h]
    """
    return np.array([rect[0]+(rect[2]-TO_REMOVE)/2, rect[1]+(rect[3]-TO_REMOVE)/2]), np.array([rect[2], rect[3]])  # 0-index

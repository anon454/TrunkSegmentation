
import numpy as np
import cv2



LABEL = {'mask':0, 'water':1, 'sky':2, 'vegetation':3}
COLOR = {'mask':[0,0,0], 'water':[255,0,0], 'sky':[255,255,0],
        'vegetation':[0,255,0]}

def gen_overlay(img, mask, label_name, alpha=0.3):

    overlay = img.copy()
    mask_col = np.zeros(img.shape).astype(np.uint8)
    mask_col[mask==LABEL[label_name]] = COLOR[label_name]
    cv2.addWeighted(mask_col, alpha, overlay, 1-alpha, 0, overlay)
    return overlay


def mask2col(mask, label_name):

    mask_col = np.zeros((mask.shape + (3,))).astype(np.uint8)
    mask_col[mask==LABEL[label_name]] = COLOR[label_name]
    return mask_col


def mask2col(mask):

    mask_col = np.zeros((mask.shape + (3,))).astype(np.uint8)
    for key,value in LABEL.items():
        label_name = key
        label_id = value
        color = COLOR[label_name]
        mask_col[mask==label_id] = color
    return mask_col

def gen_overlay(img, mask, alpha=0.3):

    overlay = img.copy()
    mask_col = mask2col(mask)
    cv2.addWeighted(mask_col, alpha, overlay, 1-alpha, 0, overlay)
    return overlay


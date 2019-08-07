

from models import model_configs
from utils.segmentor import Segmentor
import utils.joint_transforms as joint_transforms
from datasets import cityscapes
from utils.misc import rename_keys_to_match

import os, glob
import numpy as np
import cv2
import pandas as pd
from scipy.interpolate import interp1d
import torch
import torchvision.transforms as standard_transforms
import time
from math import pi,cos,sin 
from pyquaternion import Quaternion

import cst


ROOT_DATA_DIR = '/mnt/lake/'
ROOT_SEG_DIR = '/home/gpu_user/assia/ws/datasets/lake/datasets/seg/'
NETWORK_FILE = 'pth/from-paper/CMU-CS-Vistas-CE.pth'
NUM_CLASS = 19

def segment(survey_id, seq_start, seq_end, iter_):
    # output dir
    out_root_dir = 'res/%d'%survey_id

    img_dir = '%s/Dataset/20%d/%d/'%(ROOT_DATA_DIR, survey_id/10000, survey_id)
    mask_dir = '%s/%d/water/auto/'%(ROOT_SEG_DIR, survey_id)
    print(survey_id)
    seq_dir_l = sorted(glob.glob('%s/00*'%img_dir))

    filenames_ims, filenames_segs = [],[]
    filenames_mask = []

    for seq_dir in seq_dir_l:
        seq = int(os.path.basename(seq_dir))
        if int(seq)<seq_start:
            continue
        if int(seq) == seq_end:
            break 
        print('\n%04d'%seq)
        seq_img_fn_l = sorted(os.listdir(seq_dir))
        img_num = len(seq_img_fn_l)

        out_dir = '%s/%04d/lab/'%(out_root_dir, seq)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        #for i, img_root_fn in enumerate(seq_img_fn_l[img_start:img_end+1]):
        for i in range(0, img_num, iter_):
            img_root_fn = seq_img_fn_l[i]
            if i%100==0:
                print('%d  %04d  %d/%d'%(survey_id, seq, i, img_num))

            img_fn = '%s/%s'%(seq_dir, img_root_fn)
            out_fn = '%s/%s.png'%(out_dir, img_root_fn.split(".")[0])
            mask_fn = '%s/%04d/%s'%(mask_dir, seq, img_root_fn)
            #print('%s\n%s'%(img_fn, out_fn))
            print('mask_fn: %s'%mask_fn)
            filenames_ims.append(img_fn)
            filenames_segs.append(out_fn)
            filenames_mask.append(mask_fn)

    # network model
    print("Using CUDA" if torch.cuda.is_available() else "Using CPU")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Network and weight loading
    model_config = model_configs.PspnetCityscapesConfig()
    net = model_config.init_network().to(device)
    print('load model ' + NETWORK_FILE)
    state_dict = torch.load(NETWORK_FILE, map_location=lambda storage, 
            loc: storage)
    # needed since we slightly changed the structure of the network in pspnet
    state_dict = rename_keys_to_match(state_dict)
    net.load_state_dict(state_dict)
    net.eval()


    # data proc
    input_transform = model_config.input_transform
    pre_validation_transform = model_config.pre_validation_transform
    # make sure crop size and stride same as during training
    sliding_crop = joint_transforms.SlidingCropImageOnly(
        713, 2/3.)


    # encapsulate pytorch model in Segmentor class
    print("Class number: %d"%net.n_classes) # 19
    segmentor = Segmentor( net, net.n_classes, colorize_fcn =
            cityscapes.colorize_mask, n_slices_per_pass = 10)

    # let's go
    count = 1
    t0 = time.time()
    for i, im_file in enumerate(filenames_ims):
        save_path = filenames_segs[i]
        mask_path = filenames_mask[i]
        tnow = time.time()
        print( "[%d/%d (%.1fs/%.1fs)] %s" % (count, len(filenames_ims), 
            tnow - t0, (tnow - t0) / count * len(filenames_ims), im_file))
        #print(save_path)

        segmentor.run_and_save( im_file, save_path, '',
                pre_sliding_crop_transform = pre_validation_transform,
                sliding_crop = sliding_crop, input_transform = input_transform,
                skip_if_seg_exists = True, use_gpu = True, save_logits=False,
                mask_path=mask_path)
        count += 1 


def get_good_pose(survey_id, seq_start, seq_end, iter_, survey_dir, pose_dir):
    # Load img list and timespamps
    survey_f = '%s/%s/image_auxilliary.csv'%(survey_dir, survey_id)
    img_aux_f = open(survey_f)
    new_names = ['t_sec','seq','x','y','theta','pan','tilt','fx','fy','cx','cy','width','height','omega','battery','rc']
    survey = pd.read_csv(img_aux_f,dtype=np.float64,comment='%',names=new_names)
    survey = survey[np.abs(survey['tilt']-0.199948)<1e-5] #delete the part before camera is ready 
    img_id_l = survey['seq'].values.astype(int) # img id list
    img_t_l = survey['t_sec'].values # img timestamps
    img_pan_l = survey['pan'].values # img camera pan
    
    
    # Load better poses than gps [(timestamps, pose)]
    optposes_fn = '%s/%s/optposes.csv'%(pose_dir, survey_id) #t,pose
    optposes_f = open(optposes_fn)
    opt_new_names = ['kf','t','x','y','theta','xg','yg','thetag']
    optposes = pd.read_csv(optposes_f,dtype=np.float64,comment='%',names=opt_new_names)
    pose_kf_l = optposes['kf'].values
    pose_t_l = optposes['t'].values
    pose_x_l = optposes['xg'].values
    pose_y_l = optposes['yg'].values
    pose_theta_l=optposes['thetag'].values
    
    # interpolation functions
    fx     = interp1d(pose_t_l, pose_x_l)
    fy     = interp1d(pose_t_l, pose_y_l)
    ftheta = interp1d(pose_t_l, pose_theta_l)

    # align timestamps between poses and img list
    t_count=0
    while img_t_l[t_count]<pose_t_l[0]:
        t_count+=1

    out_pose_l = []
    out_img_id_l = []
    for i in range(t_count, len(img_t_l), iter_):
        img_id = img_id_l[i]
        seq = int(img_id / 1000)
        if seq < seq_start:
            continue
        if seq > seq_end:
            break

        t = img_t_l[i]
        # check exact timestamp of img is within nice pose timestamps for interpolation
        if t<pose_t_l[0] or t>pose_t_l[-1]:
            continue
    
        # interpolate camera pose from the nice poses
        x, y, theta = fx(t), fy(t), ftheta(t)
        rot = theta

        # transformation from boat to world frame
        R_w_b = np.array([
            [np.cos(rot), -np.sin(rot), 0],
            [np.sin(rot), np.cos(rot), 0],
            [0,             0,          1]])
        t_w_b = np.array([x,y,0]) # t: world -> camera

        T_w_b = np.eye(4)
        T_w_b[:3,:3] = R_w_b
        T_w_b[:3,3] =  t_w_b

        T_b_w = np.linalg.inv(T_w_b)
        R_b_w = T_b_w[:3,:3]
        t_b_w = T_b_w[:3,3]
        
   
        # transformation from boat to camera
        R_c_b = np.array([
            [-1, 0, 0],
            [0, 0, -1],
            [0, -1, 0]])
        t_c_b = 0
        T_c_b = np.eye(4)
        T_c_b[:3,:3] = R_c_b
        T_c_b[:3,3] = t_c_b

        T_b_c = np.linalg.inv(T_c_b)

        T_c_w = np.dot(T_c_b, T_b_w)
        R_c_w = T_c_w[:3,:3]
        t_c_w = T_c_w[:3,3]

        T_w_c = np.linalg.inv(T_c_w)
        
        qw, qx, qy, qz = Quaternion(matrix=R_b_w) # OK
        
        out_pose_l.append([img_id, T_w_c])
    return out_pose_l



def segment_across_season(survey0_id, survey1_id, seq_start, seq_end, iter_):

    # sample images in survey0 then find nearest one in survey1

    # sample images in survey0 with their good pose
    pose0_l = get_good_pose(survey0_id, seq_start, seq_end, iter_, 
            cst.SURVEY_DIR, cst.POSE_DIR)
    #for pose in pose0_l:
    #    print(pose)
    #    break
    #exit(0)
    # sample all good poses for survey 1
    pose1_l = get_good_pose(survey0_id, seq_start, seq_end, 10,
            cst.SURVEY_DIR, cst.POSE_DIR)
    #for pose in pose1_l:
    #    print(pose)


    # for each img in survey0_id, find the nearest one in survey1_id
    #for pose in pose0_l:
    #    print(pose)
    #    print(pose[1])
    #    input('wait')
    t0 = np.array([pose[1][:3,3] for pose in pose0_l])
    t1 = np.array([pose[1][:3,3] for pose in pose1_l])
    q0 = np.array([Quaternion(matrix=pose[1][:3,:3]) for pose in pose0_l])
    q1 = np.array([Quaternion(matrix=pose[1][:3,:3]) for pose in pose1_l])

    t0 = np.expand_dims(t0, 1)
    t1 = np.expand_dims(t1, 0)
    d_t = np.linalg.norm(t0 - t1, ord=None, axis=2)

    match01 = np.argmin(d_t, axis=1)

    
    # display matches
    img0_dir = '%s/%d/'%(cst.SURVEY_DIR, survey0_id)
    img1_dir = '%s/%d/'%(cst.SURVEY_DIR, survey1_id)

    for i, pose in enumerate(pose0_l):
        img0_id = pose[0]
        img1_id = pose1_l[match01[i]][0]

        img0_fn = '%s/%04d/%04d.jpg'%(img0_dir, img0_id/1000, img0_id%1000)
        img1_fn = '%s/%04d/%04d.jpg'%(img1_dir, img1_id/1000, img1_id%1000)
        print('%s\n%s\n'%(img0_fn, img1_fn))

        img0, img1 = cv2.imread(img0_fn), cv2.imread(img1_fn)
        cv2.imshow('img0', img0)
        cv2.imshow('img1', img1)
        cv2.waitKey(0)




     
    



if __name__ == '__main__':
    survey_id = 150429
    seq_start = 2
    seq_end = 32

    survey_id = 150216
    seq_start = 1
    seq_end = 40

    iter_ = 100
    #segment(survey_id, seq_start, seq_end, iter_)


    survey0_id = 150429
    survey1_id = 150216
    segment_across_season(survey0_id, survey1_id, seq_start, seq_end, iter_)




import os
import cv2
import numpy as np
import h5py

#fn = 'correspondence_run1_overcast-reference_run2_dawn_c13_c2645.mat' # robotcar

DATA_ROOT_DIR = '/home/abenbihi/ws/datasets/'

DATA = 1

if (DATA == 0):
    corr_dir = '%s/correspondence/Robotcar/correspondence_data'%DATA_ROOT_DIR
    root_imgs = '%s/RobotCar-Seasons/images/'%DATA_ROOT_DIR
elif (DATA == 1):
    SLICE = 6
    corr_dir = '%s/correspondence/CMU/correspondence_data/slice%d/'%(DATA_ROOT_DIR,SLICE)
    root_imgs = '%s/CMU-Seasons/images/'%DATA_ROOT_DIR
else:
    print("Error: Get you MTF DATA macro ok.")
    exit(1)


for fn in sorted(os.listdir(corr_dir)):

    #fn = 'correspondence_slice6_run13_run21_c11_c21.mat' # cmu
    #fn = 'correspondence_run1_overcast-reference_run2_dawn_c13_c2645.mat' # robotcar
    #f = h5py.File('%s'%(fn), 'r')
    
    mat_content = {}
    f = h5py.File('%s/%s'%(corr_dir, fn), 'r')
    for k, v in f.items():
        asd = 0
        mat_content[k] = np.array(v)
    
    im1name = ''.join(chr(a)
                      for a in mat_content['im_i_path'])  # convert to string
    im2name = ''.join(chr(a)
                      for a in mat_content['im_j_path'])  # convert to string
    mat_content['pt_i'] = np.swapaxes(mat_content['pt_i'], 0, 1)
    mat_content['pt_j'] = np.swapaxes(mat_content['pt_j'], 0, 1)
    mat_content['dist_from_center'] = np.swapaxes(
        mat_content['dist_from_center'], 0, 1)

    slice_ = fn.split("_")[1]
    
    
    suffix = im1name.split("/")[-1]
    suffix = suffix.split(".")[0] + '_rect.jpg'
    img1path = '%s/%s/database/%s'%(root_imgs, slice_, suffix)
    if not os.path.exists(img1path):
        img1path = '%s/%s/query/%s'%(root_imgs, slice_, suffix)
        if not os.path.exists(img1path):
            continue

    suffix = im2name.split("/")[-1]
    suffix = suffix.split(".")[0] + '_rect.jpg'
    img2path = '%s/%s/database/%s'%(root_imgs, slice_, suffix)
    if not os.path.exists(img2path):
        img2path = '%s/%s/query/%s'%(root_imgs, slice_, suffix)
        if not os.path.exists(img2path):
            continue

    print(fn)
    print(im1name) 
    print(img1path)
    print(img2path)
    
    img1 = cv2.imread(img1path)
    img2 = cv2.imread(img2path)
    cv2.imshow('img', np.hstack((img1, img2)))
    k = cv2.waitKey(0) & 0xFF
    if k == ord("q"):
        exit(0)

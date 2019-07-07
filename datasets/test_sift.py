
import os
import cv2
import numpy as np
import h5py

method = 'sift'


max_num_feat = -1

# feature extraction handler
if method=='sift':
    if max_num_feat != -1:
        fe = cv2.xfeatures2d.SIFT_create(max_num_feat)
    else:
        fe = cv2.xfeatures2d.SIFT_create()
elif method=='surf':
    fe = cv2.xfeatures2d.SURF_create(400)
elif method=='orb':
    fe = cv2.ORB_create()
elif method=='mser':
    fe = cv2.MSER_create()
elif method=='akaze':
    fe = cv2.AKAZE_create()
else:
    print('This mtf method is not handled: %s'%method)
    exit(1)

# feature matcher handler for visualization
if method == 'orb':
    norm = 'hamming'
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
elif method=='akaze':
    norm = 'hamming'
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
else:
    norm = 'L2'
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params,search_params)

#fn = 'correspondence_run1_overcast-reference_run2_dawn_c13_c2645.mat' # robotcar

DATA_ROOT_DIR = '/home/abenbihi/ws/datasets/'

DATA = 1

if (DATA == 0):
    corr_dir = '%s/correspondence/Robotcar/correspondence_data'%DATA_ROOT_DIR
    root_imgs = '%s/RobotCar-Seasons/images/'%DATA_ROOT_DIR
elif (DATA == 1):
    SLICE = 7
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
    
    img0 = cv2.imread(img1path)
    img1 = cv2.imread(img2path)
    cv2.imshow('img', np.hstack((img0, img1)))
    
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    # detect and describe
    kp0, des0 = fe.detectAndCompute(img0,None)
    if max_num_feat != -1:
        kp0 = kp0[:max_num_feat]
        des0 = des0[:max_num_feat, :]
    kp_on_img0 = np.tile(np.expand_dims(img0,2), (1,1,3))
    # draw kp on img
    for i,kp in enumerate(kp0):
        pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
        cv2.circle(kp_on_img0, pt, 1, (0, 255, 0), -1, lineType=16)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # detect and describe
    kp1, des1 = fe.detectAndCompute(img1,None)
    if max_num_feat != -1:
        kp1 = kp1[:max_num_feat]
        des1 = des1[:max_num_feat, :]
    # draw kp
    kp_on_img1 = np.tile(np.expand_dims(img1,2), (1,1,3))
    for i,kp in enumerate(kp1):
        pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
        cv2.circle(kp_on_img1, pt, 1, (0, 255, 0), -1, lineType=16)

    good = []
    if method=='orb':
        matches = matcher.match(des0,des1)
        # Sort them in the order of their distance.
        # these are not necessarily good matches, I just called them
        # good to be homogeneous
        good = sorted(matches, key = lambda x:x.distance)
    else:
        matches = matcher.knnMatch(des0, des1,k=2)
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                good.append(m)

    match_des_img = cv2.drawMatches(img0, kp0, img1, kp1, good, None, 
            flags=2)

    cv2.imshow('match_des', match_des_img)
    cv2.imshow('kp_on', np.hstack((kp_on_img0, kp_on_img1)))
    k = cv2.waitKey(0) & 0xFF
    if k == ord("q"):
        exit(0)



import os, glob
import numpy as np
import cv2


ROOT_DATA_DIR = '/mnt/lake/'
def cmp_mask(survey_id, seq_start, seq_end):

    out0_root_dir = 'res/%d_0'%survey_id
    out1_root_dir = 'res/%d_1'%survey_id
    out_root_dir = 'res/%d'%survey_id

    img_dir = '%s/Dataset/20%d/%d/'%(ROOT_DATA_DIR, survey_id/10000, survey_id)
    print(survey_id)
    seq_dir_l = sorted(glob.glob('%s/00*'%img_dir))

    filenames_ims, filenames_segs = [],[]

    for seq_dir in seq_dir_l:
        seq = int(os.path.basename(seq_dir))
        if int(seq)<seq_start:
            continue
        if int(seq) == seq_end:
            break 
        print('\n%04d'%seq)
        seq_img_fn_l = sorted(os.listdir(seq_dir))
        img_num = len(seq_img_fn_l)

        out0_dir = '%s/%04d/lab/'%(out0_root_dir, seq)
        out1_dir = '%s/%04d/lab/'%(out1_root_dir, seq)
        out_dir = '%s/%04d/lab/'%(out_root_dir, seq)
        
        for out_root_fn in sorted(os.listdir(out_dir)):
            
            img_fn = '%s/%s.jpg'%(seq_dir, out_root_fn.split(".")[0])
            out0_fn = '%s/%s'%(out0_dir, out_root_fn)
            out1_fn = '%s/%s'%(out1_dir, out_root_fn)
            out_fn = '%s/%s'%(out_dir, out_root_fn)
            
            print('%s\n%s'%(img_fn, out0_fn))
            img = cv2.imread(img_fn)
            out0 = cv2.imread(out0_fn)
            
            img1 = cv2.imread(img_fn)[:,:700]
            out1 = cv2.imread(out1_fn)
            out = cv2.imread(out_fn)
            
            overlay = img.copy()
            alpha = 0.3
            cv2.addWeighted(out0, alpha, overlay, 1-alpha, 0, overlay)
 
            overlay1 = img1.copy()
            alpha = 0.3
            cv2.addWeighted(out1, alpha, overlay1, 1-alpha, 0, overlay1)
 
            overlay = img1.copy()
            alpha = 0.3
            cv2.addWeighted(out, alpha, overlay, 1-alpha, 0, overlay)
          
            cv2.imshow('out0', np.hstack((img, out0, overlay)))
            cv2.imshow('out1', np.hstack((img, out1, overlay1)))
            cv2.imshow('out', np.hstack((img, out, overlay)))
            k = cv2.waitKey(0) & 0xFF
            if k == ord("q"):
                exit(0)


def show(survey_id, seq_start, seq_end):

    out_root_dir = 'res/%d'%survey_id

    img_dir = '%s/Dataset/20%d/%d/'%(ROOT_DATA_DIR, survey_id/10000, survey_id)
    print(survey_id)
    seq_dir_l = sorted(glob.glob('%s/00*'%img_dir))

    filenames_ims, filenames_segs = [],[]

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
        
        for out_root_fn in sorted(os.listdir(out_dir)):
            
            img_fn = '%s/%s.jpg'%(seq_dir, out_root_fn.split(".")[0])
            out_fn = '%s/%s'%(out_dir, out_root_fn)
            
            print('%s\n%s'%(img_fn, out_fn))
            img = cv2.imread(img_fn)[:,:700]
            out = cv2.imread(out_fn)
             
            overlay = img.copy()
            alpha = 0.3
            cv2.addWeighted(out, alpha, overlay, 1-alpha, 0, overlay)
          
            cv2.imshow('out', np.hstack((img, out, overlay)))
            k = cv2.waitKey(0) & 0xFF
            if k == ord("q"):
                exit(0)




if __name__=='__main__':
    survey_id = 150429
    seq_start = 2
    seq_end = 32

    survey_id = 150216
    seq_start = 9
    seq_end = 40

    show(survey_id, seq_start, seq_end)




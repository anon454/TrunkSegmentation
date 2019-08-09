
import os, glob
import numpy as np
import cv2

import cst
import palette

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



def show_across_season(survey0_id, survey1_id):

    out_root_dir = 'res/%d_%d'%(survey1_id, survey0_id)

    img0_dir = '%s/%d/'%(cst.SURVEY_DIR, survey0_id)
    img1_dir = '%s/%d/'%(cst.SURVEY_DIR, survey1_id)

    mask0_dir = '%s/%d/water/auto/'%(cst.SEG_DIR, survey0_id)
    mask1_dir = '%s/%d/water/auto/'%(cst.SEG_DIR, survey1_id)

    seq_l = sorted(os.listdir(out_root_dir))
    for seq in seq_l:
        out_dir = '%s/%s/'%(out_root_dir, seq)
        out_fn_l = sorted(os.listdir(out_dir))
        out_fn_v = np.reshape(np.array(out_fn_l), (int(len(out_fn_l)/2), 2))
        print(out_fn_v)

        for l in out_fn_v:
            img0_id = int(l[0].split("_")[2].split(".")[0])
            seq0 = int(img0_id/1000)
            img0_fn = '%s/%04d/%04d.jpg'%(img0_dir, seq0, img0_id%1000)
            out0_fn = '%s/%s'%(out_dir, l[0])
            mask0_fn = '%s/%04d/%04d.jpg'%(mask0_dir, seq0, img0_id%1000)
            
            
            img1_id = int(l[1].split("_")[2].split(".")[0])
            seq1 = int(img1_id/1000)
            img1_fn = '%s/%04d/%04d.jpg'%(img1_dir, seq1, img1_id%1000)
            out1_fn = '%s/%s'%(out_dir, l[1])
            mask1_fn = '%s/%04d/%04d.jpg'%(mask1_dir, seq1, img1_id%1000)



            #print('%s\n%s'%(img0_fn, out0_fn))
            if not os.path.exists(img0_fn):
                continue
            img0 = cv2.imread(img0_fn)[:,:700]
            out0 = cv2.imread(out0_fn)
            overlay0 = img0.copy()
            alpha = 0.3
            cv2.addWeighted(out0, alpha, overlay0, 1-alpha, 0, overlay0)
            
            #print('%s\n%s'%(img1_fn, out1_fn))
            img1 = cv2.imread(img1_fn)[:,:700]
            out1 = cv2.imread(out1_fn)
            overlay1 = img1.copy()
            alpha = 0.3
            cv2.addWeighted(out1, alpha, overlay1, 1-alpha, 0, overlay1)
            

            cv2.imshow('out0', np.hstack((img0, out0, overlay0)))
            cv2.imshow('out1', np.hstack((img1, out1, overlay1)))


            k = cv2.waitKey(0) & 0xFF
            if k == ord("q"):
                exit(0)


def col2mask(out, color_l):
    num_class = len(color_l)
    mask = np.zeros(out.shape[:2]).astype(np.uint8)
    print(out.shape)
    for i, color in enumerate(color_l):
        color = [color[2], color[1], color[0]]
        toto = np.array(out==color).astype(np.int)
        #cv2.imshow('toto', (255*toto).astype(np.uint8))
        #cv2.waitKey(0)
        mask[toto[:,:,0]==1] = i
    return mask


def mask2col(mask, color_l):
    col = np.zeros(mask.shape + (3,)).astype(np.uint8)
    for i, color in enumerate(color_l):
        color = [color[2], color[1], color[0]]
        col[mask==i] = color

    #cv2.imshow('col', col)
    #cv2.waitKey(0)
    return col


def fuck(survey_id, seq_start, seq_end):

    out_root_dir = 'res/%d'%survey_id

    img_dir = '%s/Dataset/20%d/%d/'%(cst.ROOT_DATA_DIR, survey_id/10000, survey_id)
    water_dir = '%s/%d/water/auto/'%(cst.SEG_DIR, survey_id)
    #print(survey_id)
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
            water_fn = '%s/%04d/%s.jpg'%(water_dir, seq,
                    out_root_fn.split(".")[0])
            out_fn = '%s/%s'%(out_dir, out_root_fn)
            #print('%s\n%s'%(img_fn, out_fn))
            #print(water_fn)
            
            img = cv2.imread(img_fn)[:,:700]
            water = cv2.imread(water_fn, cv2.IMREAD_UNCHANGED)
            out = cv2.imread(out_fn)

            mask = col2mask(out, palette.palette)
            out[water==cst.LABEL['water']] = 0
            #mask[water==cst.LABEL['water']] = 19
            #col = mask2col(mask, palette.palette)
            #print(np.unique(water))
            #col[water==1] = 0
            #cv2.imshow('col', col)
            cv2.imshow('water', water)
            cv2.imshow('out', out)

            sky_label = 10
            veg_label = 8
            wat_label = 19
            rod_label = 0
            img[mask==wat_label] = 0
            img[mask==sky_label] = 0
            img[mask==rod_label] = 0
            #img[mask!=veg_label] = 0
            cv2.imshow('img', img)

            #overlay = img.copy()
            #alpha = 0.3
            #cv2.addWeighted(out, alpha, overlay, 1-alpha, 0, overlay)
          
            #cv2.imshow('out', np.hstack((img, out, overlay)))
            k = cv2.waitKey(0) & 0xFF
            if k == ord("q"):
                exit(0)



if __name__=='__main__':
    survey_id = 150429
    seq_start = 2
    seq_end = 32
    #fuck(survey_id, seq_start, seq_end)

    survey_id = 150216
    seq_start = 9
    seq_end = 40

    #show(survey_id, seq_start, seq_end)

    survey1_id, survey0_id = 150429, 150216
    show_across_season(survey0_id, survey1_id)




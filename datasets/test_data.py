
import numpy as np
import cv2

ignore_label = 255


#labels = [
#    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#,
#    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
#    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
#    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
#    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
#    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
#    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
#    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
#    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
#    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
#    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
#    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
#    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
#    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
#    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
#    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
#    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
#    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
#    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
#    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
#]
#


label_name = [
    'road'                 ,
    'sidewalk'             ,
    'building'             ,
    'wall'                 ,
    'fence'                ,
    'pole'                 ,
    'traffic light'        ,
    'traffic sign'         ,
    'vegetation'           ,
    'terrain'              ,
    'sky'                  ,
    'person'               ,
    'rider'                ,
    'car'                  ,
    'truck'                ,
    'bus'                  ,
    'train'                ,
    'motorcycle'           ,
    'bicycle'              ,
    'mask' # added by Assia
]

# rgb
palette = [[128, 64, 128], 
        [244, 35, 232], 
        [70, 70, 70], 
        [102, 102, 156], 
        [190, 153, 153], 
        [153, 153, 153], 
        [250, 170, 30],
        [220, 220, 0], 
        [107, 142, 35], 
        [152, 251, 152], 
        [70, 130, 180], 
        [220, 20, 60], 
        [255, 0, 0], 
        [0, 0, 142], 
        [0, 0, 70],
        [0, 60, 100], 
        [0, 80, 100], 
        [0, 0, 230], 
        [119, 11, 32],
        [0, 0, 0]] # class 20: ignored, I added it, not cityscapes


palette_bgr = [ [l[2], l[1], l[0]] for l in palette]


def lab2col(lab, colors):
    """
    Convert label map to color map
    """
    col = np.zeros((lab.shape + (3,))).astype(np.uint8)
    labels = np.unique(lab)

    if np.max(labels) >= len(colors):
        print("Error: you need more colors np.max(labels) >= len(colors): %d >= %d"
                %(np.max(labels), len(colors)) )
        exit(1)

    for label in labels:
        col[lab==label,:] = colors[label]
    return col

if __name__=='__main__':

    mask = np.random.randint(0, 19, (200, 200)).astype(np.uint8)
    mask = np.ones((200, 200), np.uint8) * 8

    col = lab2col(mask, palette_bgr)

    cv2.imshow('col', col)
    cv2.waitKey(0)

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import torch
import random

# def drawMatch():
#
#     return 0

def match_visual(img0,img1,points0,points1):
    newWidth = img0.shape[1] + img1.shape[1]
    newHeight = img0.shape[0]
    allpic = np.zeros((newHeight, newWidth, 3), np.uint8)
    allpic[:, 0:img0.shape[1]] = img0
    allpic[:, img0.shape[1]:newWidth] = img1
    for i in range(len(points0)):
        a = random.randint(0, 255)
        b = random.randint(0, 255)
        c = random.randint(0, 255)
        allpic = cv2.circle(allpic, (points0[i][0], points0[i][1]), 6, (a, b, c), -1)
        allpic = cv2.circle(allpic, (points1[i][0] + img0.shape[1], points1[i][1]), 6,
                            (a, b, c), -1)
        allpic = cv2.line(allpic, (points0[i][0], points0[i][1]),(points1[i][0] + img0.shape[1], points1[i][1]),(a, b, c), 1)
    plt.imshow(allpic)
    plt.show()
    return 0


# or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()   # load the extractor
matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1,filter_threshold=0.1).eval().cuda()   # load the matcher
#自己训练的：mega & 2dpic & ok

# load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
image0 = load_image('/media/zhaoyibin/3DRE/LGDATA/Satellite/data/color/color_0001.png').cuda()
image1 = load_image('/media/zhaoyibin/3DRE/LGDATA/Satellite/data/color/color_0002.png').cuda()

img0 = cv2.imread('/media/zhaoyibin/3DRE/LGDATA/Satellite/data/color/color_0001.png')
img1 = cv2.imread('/media/zhaoyibin/3DRE/LGDATA/Satellite/data/color/color_0002.png')




# extractor = SuperPoint(max_num_keypoints=128).eval().cuda()  # load the extractor
# matcher = LightGlue(features='zybtest').eval().cuda()  # load the matcher

# or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
# extractor = DISK(max_num_keypoints=2048).eval().cuda()  # load the extractor
# matcher = LightGlue(features='disk').eval().cuda()  # load the matcher

# load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]

# extract local features
feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
feats1 = extractor.extract(image1)


# keypoints = feats0['keypoints'][0]
# keypoints = keypoints.tolist()
# for i in range(2048):
#     show = cv2.circle(img0, (int(keypoints[i][0]), int(keypoints[i][1])), 6, (0, 0, 1), -1)
# plt.imshow(show)


# match the features
matches01 = matcher({'image0': feats0, 'image1': feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
matches = matches01['matches']  # indices with shape (K,2)
points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
points0 = np.int32(points0.cpu())
points1 = np.int32(points1.cpu())
match_visual(img0, img1,points0,points1)



from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import torch

# def drawMatch():
#
#     return 0

def match_visual(img0,img1,points0,points1,matches):
    newWidth = img0.shape[1] + img1.shape[1]
    newHeight = img0.shape[0]
    allpic = np.zeros((newHeight, newWidth, 3), np.uint8)
    allpic[:, 0:img0.shape[1]] = img0
    allpic[:, img0.shape[1]:newWidth] = img1
    for i in range(len(matches)):
        allpic = cv2.circle(allpic, (points0[matches[i][0]][0], points0[matches[i][0]][1]), 3, (255, 0, 0), -1)
        allpic = cv2.circle(allpic, (points1[matches[i][1]][0] + img0.shape[1], points1[matches[i][1]][1]), 3, (255, 0, 0), -1)
    # plt.imshow(allpic)
    # plt.show()

    matches_np = np.array(matches.tolist())
    for i in range(len(matches)):
        allpic = cv2.line(allpic, (points0[matches[i][0]][0], points0[matches[i][0]][1]), (points1[matches[i][1]][0] + img0.shape[1], points1[matches[i][1]][1]),
                          (255, 0, 0), 1)
    plt.imshow(allpic)
    plt.show()
    return 0


# or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
extractor = SuperPoint(max_num_keypoints=512).eval().cuda()   # load the extractor
matcher = LightGlue(features='superpoint',depth_confidence=0.9, width_confidence=0.95).eval().cuda()   # load the matcher

# load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
image0 = load_image('/home/zhaoyibin/3DRE/sfm-learn/SFM_OWN/DATA/dtu/scan10/images/00000020.jpg').cuda()
image1 = load_image('/home/zhaoyibin/3DRE/sfm-learn/SFM_OWN/DATA/dtu/scan10/images/00000021.jpg').cuda()

img0 = cv2.imread('/home/zhaoyibin/3DRE/sfm-learn/SFM_OWN/DATA/dtu/scan10/images/00000020.jpg')
img1 = cv2.imread('/home/zhaoyibin/3DRE/sfm-learn/SFM_OWN/DATA/dtu/scan10/images/00000021.jpg')




extractor.extract(image0)

start = time.time()

# extract local features
feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
feats1 = extractor.extract(image1)

end = time.time()
print("运行时间=",end-start,"s")


# match the features
start1 = time.time()

matches01 = matcher({'image0': feats0, 'image1': feats1})

end1 = time.time()
print("运行时间=",end1-start1,"s")

feats00, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
matches = matches01['matches']  # indices with shape (K,2)
points0 = feats0['keypoints']  # coordinates in image #0, shape (K,2)
points0 = np.array(points0.cpu()).astype(int)
points1 = feats1['keypoints'] # coordinates in image #1, shape (K,2)
points1 = np.array(points1.cpu()).astype(int)
matches = np.array(matches.cpu())

match_visual(img0, img1,points0,points1,matches)



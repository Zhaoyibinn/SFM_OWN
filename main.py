import numpy as np
from feature import SURF_opencv
from sfm import SFM
import open3d as o3d
from my_math import my_rotation
import os
def main(imgpaths,camK,imgnum):
    SURF_extraction = SURF_opencv.SURF_cv2(imgpaths, imgnum)
    SURF_extraction.surf_feature_extraction()
    sfm_test = SFM.sfm_cv2(SURF_extraction, camK)
    pcd_np = sfm_test.reconstruct_allpics()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    MinPts = 5  # 邻域球内的最少点个数，小于该个数为噪声点
    r = 1
    pcda, idx = pcd.remove_radius_outlier(MinPts, r)
    o3d.visualization.draw_geometries([pcda])
    return 0

if __name__=="__main__":
    img1='DATA/dtu/scan10/images/00000021.jpg'
    img2='DATA/dtu/scan10/images/00000022.jpg'
    img3='DATA/dtu/scan10/images/00000023.jpg'
    img4 = 'DATA/dtu/scan10/images/00000024.jpg'
    imgnum=4
    imgpaths = [img1,img2,img3,img4]
    # Dir_path = "./DATA/dtu/scan10/images/"
    # imgpaths = os.listdir(Dir_path)
    # imgpaths.sort()
    # imgnum = len(imgpaths)
    # for i in range(imgnum):
    #     imgpaths[i] = Dir_path + imgpaths[i]
    camK = np.array([[2892.33, 0.0, 823.206],
                     [0.0, 2883.18, 619.07],
                     [0.0, 0.0, 1.0]])
    main(imgpaths,camK,imgnum)
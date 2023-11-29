import numpy as np
from feature import SURF_opencv
from sfm import SFM
import open3d as o3d
from my_math import rotation_opencv
def main(imgpaths,camK,imgnum):
    SURF_extraction = SURF_opencv.SURF_cv2(imgpaths, imgnum)







    list_kp_1s, list_kp_2s, matchidxs = SURF_extraction.surf_feature_extraction()
    # 处理0号，1号两张图片
    list_kp0, list_kp1, _ = SURF_extraction.get_feature(0, 1)
    R01, t01, points3d01 = SFM.proc_2_pics(camK, list_kp0, list_kp1)
    co_feature_idx, co_feature01_pic0_xy, co_feature01_pic1_xy, co_feature12_pic2_xy = SURF_extraction.get_co_feature(0, 1, 2)
    # create_co_see_pic(list_kp_1s, list_kp_2s, matchidxs,imgnum)
    # 从0号，1号图片重建得到的点云中，按索引挑选出共视特征点的点云
    co_points3d_01_cam0 = points3d01[co_feature_idx, :]
    # 增量式构建2号图片与1号图片进行三角化的点云
    list_kp1, list_kp2, _ = SURF_extraction.get_feature(1, 2)
    points3d_12_cam0, R02, t02 = SFM.increment_sfm(co_points3d_01_cam0, co_feature12_pic2_xy, camK, R01, t01, list_kp1, list_kp2)

    # 处理1号，2号两张图片
    list_kp2, list_kp3, _ = SURF_extraction.get_feature(2, 3)
    co_feature_idx, co_feature12_pic1_xy, co_feature12_pic2_xy, co_feature23_pic3_xy = SURF_extraction.get_co_feature(1, 2, 3)
    # 从1号，2号图片重建得到的点云中，按索引挑选出共视特征点的点云
    co_points3d_12_cam0 = points3d_12_cam0[co_feature_idx, :]
    points3d_23_cam0, R03, t03 = SFM.increment_sfm(co_points3d_12_cam0, co_feature23_pic3_xy, camK, R02, t02, list_kp2, list_kp3)

    pcd01 = o3d.geometry.PointCloud()
    pcd01.points = o3d.utility.Vector3dVector(points3d01)
    pcd01.paint_uniform_color([0, 0, 1])

    pcd12 = o3d.geometry.PointCloud()  # 展示三维点
    pcd12.points = o3d.utility.Vector3dVector(points3d_12_cam0)
    pcd12.paint_uniform_color([1, 0, 0])

    pcd23 = o3d.geometry.PointCloud()
    pcd23.points = o3d.utility.Vector3dVector(points3d_23_cam0)
    pcd23.paint_uniform_color([0, 1, 0])

    pcd01=pcd01+pcd12+pcd23

    MinPts = 5  # 邻域球内的最少点个数，小于该个数为噪声点
    R = 1
    pcd01, idx = pcd01.remove_radius_outlier(MinPts, R)

    o3d.visualization.draw_geometries([pcd01])


    return 0

if __name__=="__main__":
    print("main")
    img1='DATA/dtu/scan10/images/00000021.jpg'
    img2='DATA/dtu/scan10/images/00000022.jpg'
    img3='DATA/dtu/scan10/images/00000023.jpg'
    img4 = 'DATA/dtu/scan10/images/00000024.jpg'
    imgnum=4
    imgpaths = [img1,img2,img3,img4]
    camK = np.array([[2892.33, 0.0, 823.206],
                     [0.0, 2883.18, 619.07],
                     [0.0, 0.0, 1.0]])
    main(imgpaths,camK,imgnum)
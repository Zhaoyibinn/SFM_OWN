import time

import numpy as np
from feature import SURF
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import networkx as nx
from estimator import triangulation, PNP_opencv
from my_math import my_rotation
from feature import SURF_opencv
from feature import my_feature_class
class sfm_cv2:
    def __init__(self, feature_extractor:my_feature_class.feature, camK):
        self.feature_extractor_ = feature_extractor
        self.camK_ = camK
    def reconstruct_allpics(self):
        img_idx_list = range(self.feature_extractor_.imgnum_)
        list_kp0, list_kp1, _ = self.feature_extractor_.get_feature(0, 1)
        R01, t01, points3d_01 = self.__proc_2_pics(self.camK_, list_kp0, list_kp1)
        g_base_to_Mid = my_rotation.Rt2SE3(R01, t01)
        points3d_old_iter = points3d_01
        points3d_camBase_list = []
        points3d_camBase_list.append(points3d_old_iter)
        for i in range(self.feature_extractor_.imgnum_ - 2):
            sub_idx_list = img_idx_list[i:i + 3]
            # 顺序取出三张图片的索引，进行增量重建
            points3d_old_iter, g_base_to_Mid= self.__reconstruct_3pics(sub_idx_list[0], sub_idx_list[1], sub_idx_list[2], g_base_to_Mid, points3d_old_iter)
            points3d_camBase_list.append(points3d_old_iter)
        pcd_all_np = np.vstack(points3d_camBase_list)
        # pcd_all_np = points3d_camBase_list
        return pcd_all_np

    def __reconstruct_3pics(self, idx_pic1:int, idx_pic2:int, idx_pic3:int, g_Base_to_Mid, points3d_12_camBase):
        '''
        给定3个图片序号（升序排列），
        返回以图片1的相机位姿为基准坐标系下，
        图片1和2重建得到的点云，
        图片2和3重建得到的点云，
        图片1的相机位姿到图片2的相机位姿的姿态变换
        图片1的相机位姿到图片3的相机位姿的姿态变换
        :param idx_pic1: 图片1的序号
        :param idx_pic2: 图片2的序号
        :param idx_pic3: 图片3的序号
        :return:
                 points3d_12_cam0：图片2和图片3恢复得到的增量点云 numpy X*3
                 g_Base_to_End: base相机位姿到图片3的相机位姿的姿态变换 numpy 4*4
        '''
        if idx_pic1 < idx_pic2 and idx_pic2 < idx_pic3:
            # 增加2号图片，提取其与0号，1号图片的共视特征
            co_feature_idx, co_feature01_pic0_xy, co_feature01_pic1_xy, co_feature12_pic2_xy = self.feature_extractor_.get_co_feature(idx_pic1, idx_pic2, idx_pic3)
            # 从0号，1号图片重建得到的点云中，按索引挑选出共视特征点的点云
            co_points3d_01_cam0 = points3d_12_camBase[co_feature_idx, :]
            # 增量式构建2号图片与1号图片进行三角化的点云
            R_Base_to_Mid, t_Base_to_Mid = my_rotation.SE32Rt(g_Base_to_Mid)
            list_kp1, list_kp2, _ = self.feature_extractor_.get_feature(idx_pic2, idx_pic3)
            points3d_12_cam0, R02, t02 , g10= self.__increment_sfm(co_points3d_01_cam0, co_feature12_pic2_xy, self.camK_, R_Base_to_Mid, t_Base_to_Mid, list_kp1, list_kp2)
            g_Base_to_End = my_rotation.Rt2SE3(R02, t02)
            return points3d_12_cam0, g_Base_to_End
        else:
            raise ValueError("输入必须增序排列")
        
        pass
    def __proc_2_pics(self, camK, list_kp1, list_kp2):
        '''
        任意两张图片处理，用对极几何求解R，t，并三角化计算得到三维点作为初始点云
        :param
        :param camK: 相机内参
        :param list_kp1: 需要处理的图片1的特征点的像素坐标
        :param list_kp2: 需要处理的图片2的特征点的像素坐标
        :return:    R：旋转矩阵；
                    t：平移向量；
                    points3d：三维坐标点 numpy 矩阵（K*3）；
        '''
        good_F, status = cv2.findFundamentalMat(list_kp1, list_kp2, method=cv2.FM_RANSAC, ransacReprojThreshold=3,
                                                confidence=0.99)  # 使用RANSAC方法计算基本矩阵，函数参考
        # https://blog.csdn.net/bb_sy_w/article/details/121082013?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170108654916800215081297%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170108654916800215081297&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-5-121082013-null-null.142^v96^pc_search_result_base2&utm_term=cv2.findFundamentalMat&spm=1018.2226.3001.4187
        print("该两张图片的基础矩阵F=", end="")
        print(good_F)
        E = np.dot(np.dot(np.transpose(camK), good_F), camK)  # 计算本质矩阵，就是(K.T)*F*K
        print("该两张图片的本质矩阵E=", end="")
        print(E)
        retval, R, t, mask = cv2.recoverPose(E, list_kp1, list_kp2, camK)  # 计算得到R，t
        print("计算得到前两张图片的R=", end="")
        print(R)
        print("计算得到前两张图片的t=", end="")
        print(t)
        points3d = triangulation.triangulate(camK, R, t, list_kp1, list_kp2)
        # calerror(camK, list_kp1, list_kp2, points3d, R, t) #计算重投影误差
        return R, t, points3d

    def __increment_sfm(self, co_points3d01, co_feature_extra_pic_xy, camK, R01, t01, list_kp1_12, list_kp2_12):
        '''
        对于新加入的图片进行增量式重建
        :param co_points3d01: 图片0和图片1构建出的点云中对应的2维特征和图片2共视的部分 numpy(Y*3)
        :param co_feature_extra_pic_xy: 图片2中共视的2维特征点的坐标
        :param camK: 相机内参
        :param R01: 图片0到图片1的旋转矩阵
        :param t01: 图片0到图片1的平移向量
        :param list_kp1_12: 图片1和图片2进行特征匹配得到的图片1中的特征点的像素坐标
        :param list_kp2_12:图片1和图片2进行特征匹配得到的图片2中的特征点的像素坐标
        :return: points3d_12_cam0：在cam0的坐标系下的增量点云 X*3
                 R02：cam0到cam2的旋转矩阵，numpy 3*3
                 t02：cam0到cam2的平移向量，numpy 3*1
        '''
        # 图片0到图片1的位姿
        transform_matrix01 = my_rotation.Rt2SE3(R01, t01)

        # 图片0到图片2（增量图片）的位姿
        # 通过PNP算法，基于共视点恢复出增量图片的相机姿态
        ok, rotation_vector02, translation_vector02 = PNP_opencv.EPNP(co_points3d01, co_feature_extra_pic_xy, camK)
        rot_mat02 = my_rotation.rotation_vector2matrix(rotation_vector02)
        transform_matrix02 = my_rotation.Rt2SE3(rot_mat02, translation_vector02)

        # 计算图片1到图片2（增量图片）之间的位姿
        transform_matrix12 = transform_matrix02 @ np.linalg.inv(transform_matrix01)

        # 使用图片1和图片2进行三角化得到增量点云
        R12, t12 = my_rotation.SE32Rt(transform_matrix12)
        # 得到的点云以图片1的相机姿态为原点坐标系
        points3d_12_cam1 = triangulation.triangulate(camK, R12, t12, list_kp1_12, list_kp2_12)

        # 计算图片1到图片0的位姿
        transform_matrix10 = np.linalg.inv(transform_matrix01)
        R10, t10 = my_rotation.SE32Rt(transform_matrix10)
        # 将增量点云统一到第0张图片的基准坐标系
        points3d_12_cam0 = (R10 @ points3d_12_cam1.T + t10).T
        R02, t02 = my_rotation.SE32Rt(transform_matrix02)
        return points3d_12_cam0, R02, t02, transform_matrix10

def create_co_see_pic(list_kp_1s,list_kp_2s,matchidxs,imgnum):
    '''
    创建共视图，即通过匹配的特征点数量确定哪些图片是两两接近的（暂时还没用到，可以可视化）
    :param list_kp_1s: 特征匹配数组中第一张图片的特征点像素坐标，N*N个格子（N为图片数量），每个格子里面存着许多对特征点
    :param list_kp_2s: 特征匹配数组中第二张图片的特征点像素坐标
    :param matchidxs: 特征匹配idx数组，N*N个格子，每个格子里面存着所有特征点对所对应的图片特征点idx
    :param imgnum: 图片数量
    :return: None
    '''
    G = nx.Graph()
    for i in range(imgnum):
        G.add_nodes_from([i])

    for i in range(imgnum):
        for n in range(i+1,imgnum):
            match_num = matchidxs[i][n].__len__()
            G.add_weighted_edges_from([(i, n, match_num)])

    # pos = nx.spring_layout(G)  # 可视化全部共识图
    # nx.draw(G, pos, with_labels=True, alpha=0.5)
    # labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    # plt.show()

    return 0

def pixel2cam(p,camK):
    campoint = [(p[0]-camK[0,2])/camK[0,0],(p[1]-camK[1,2])/camK[1,1]]
    return campoint


def calerror(camK,list_kp1, list_kp2,points3d,R,t):
    '''
    用于计算三角化的重投影误差，参考高翔视觉里程计1，此处暂时不用，保留
    :param camK: 内参矩阵
    :param list_kp1: 第一张图的像素坐标
    :param list_kp2: 第二张图的像素坐标
    :param points3d: 三维坐标
    :param R:
    :param t:
    :return: None
    '''
    K=camK
    i=0
    while i < list_kp1.size/2:
    #
        print("正在计算点",end="")
        print(i)
        pt1_cam = pixel2cam(list_kp1[i],camK);
        pt1_cam_3d = [points3d[i][0] / points3d[i][2],points3d[i][1] / points3d[i][2] ]
        print( "point in the first camera frame: ",end="")
        print( pt1_cam )
        print("point projected from 3D ", end="")
        print(pt1_cam_3d)
        print(" ")
    #
    # // 第二个图
        pt2_cam = pixel2cam(list_kp2[i], camK);
        pt2_trans = np.dot(R,points3d[i].T)+t.T[0]
        pt2_trans=pt2_trans/pt2_trans[2]
        print("point in the second camera frame: ", end="")
        print(pt2_cam)
        print("point projected from 3D ", end="")
        print(pt2_trans)
        print(" ")
        print(" ")
        print("误差=", end="")
        i=i+1
    # Mat
    # pt2_trans = R * (Mat_ < double > (3, 1) << points[i].x, points[i].y, points[i].z) + t;
    # pt2_trans /= pt2_trans.at < double > (2, 0);
    # cout << "point in the second camera frame: " << pt2_cam << endl;
    # cout << "point reprojected from second frame: " << pt2_trans.t() << endl;
    # cout << endl;
    # }
    return 0





def proc_2_pics(camK, list_kp1, list_kp2):
    '''
    任意两张图片处理，用对极几何求解R，t，并三角化计算得到三维点作为初始点云
    :param
    :param camK: 相机内参
    :param list_kp1: 需要处理的图片1的特征点的像素坐标
    :param list_kp2: 需要处理的图片2的特征点的像素坐标
    :return:    R：旋转矩阵；
                t：平移向量；
                points3d：三维坐标点 numpy 矩阵（K*3）；
    '''
    good_F, status = cv2.findFundamentalMat(list_kp1, list_kp2, method=cv2.FM_RANSAC, ransacReprojThreshold=1,
                                            confidence=0.99)  # 使用RANSAC方法计算基本矩阵，函数参考
    # https://blog.csdn.net/bb_sy_w/article/details/121082013?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170108654916800215081297%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170108654916800215081297&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-5-121082013-null-null.142^v96^pc_search_result_base2&utm_term=cv2.findFundamentalMat&spm=1018.2226.3001.4187
    print("该两张图片的基础矩阵F=", end="")
    print(good_F)
    E = np.dot(np.dot(np.transpose(camK), good_F), camK)  # 计算本质矩阵，就是(K.T)*F*K
    print("该两张图片的本质矩阵E=", end="")
    print(E)
    retval, R, t, mask = cv2.recoverPose(E, list_kp1, list_kp2, camK)  # 计算得到R，t
    print("计算得到前两张图片的R=", end="")
    print(R)
    print("计算得到前两张图片的t=", end="")
    print(t)
    points3d = triangulation.triangulate(camK, R, t, list_kp1, list_kp2)
    # calerror(camK, list_kp1, list_kp2, points3d, R, t) #计算重投影误差
    return R, t, points3d

def increment_sfm(co_points3d01, co_feature_extra_pic_xy, camK, R01, t01, list_kp1_12, list_kp2_12):
    '''
    对于新加入的图片进行增量式重建
    :param co_points3d01: 图片0和图片1构建出的点云中对应的2维特征和图片2共视的部分 numpy(Y*3)
    :param co_feature_extra_pic_xy: 图片2中共视的2维特征点的坐标
    :param camK: 相机内参
    :param R01: 图片0到图片1的旋转矩阵
    :param t01: 图片0到图片1的平移向量
    :param list_kp1_12: 图片1和图片2进行特征匹配得到的图片1中的特征点的像素坐标
    :param list_kp2_12:图片1和图片2进行特征匹配得到的图片2中的特征点的像素坐标
    :return: points3d_12_cam0：在cam0的坐标系下的增量点云 X*3
             R02：cam0到cam2的旋转矩阵，numpy 3*3
             t02：cam0到cam2的平移向量，numpy 3*1
    '''
    # 图片0到图片1的位姿
    transform_matrix01 = my_rotation.Rt2SE3(R01, t01)

    # 图片0到图片2（增量图片）的位姿
    # 通过PNP算法，基于共视点恢复出增量图片的相机姿态
    ok, rotation_vector02, translation_vector02 = PNP_opencv.EPNP(co_points3d01, co_feature_extra_pic_xy, camK)
    rot_mat02 = my_rotation.rotation_vector2matrix(rotation_vector02)
    transform_matrix02 = my_rotation.Rt2SE3(rot_mat02, translation_vector02)

    # 计算图片1到图片2（增量图片）之间的位姿
    transform_matrix12 = transform_matrix02 @ np.linalg.inv(transform_matrix01)

    # 使用图片1和图片2进行三角化得到增量点云
    R12, t12 = my_rotation.SE32Rt(transform_matrix12)
    # 得到的点云以图片1的相机姿态为原点坐标系
    points3d_12_cam1 = triangulation.triangulate(camK, R12, t12, list_kp1_12, list_kp2_12)

    # 计算图片1到图片0的位姿
    transform_matrix10 = np.linalg.inv(transform_matrix01)
    R10, t10 = my_rotation.SE32Rt(transform_matrix10)
    # 将增量点云统一到第0张图片的基准坐标系
    points3d_12_cam0 = (R10 @ points3d_12_cam1.T + t10).T
    R02, t02 = my_rotation.SE32Rt(transform_matrix02)
    return points3d_12_cam0, R02, t02


import numpy as np
import cv2

def triangulate(camK, R, t ,points1,points2):
    '''
    三角化
    :param camK: 相机内参矩阵 numpy (3*3)
    :param R: 旋转矩阵 numpy (3*3)
    :param t: 平移向量 numpy (3*3)
    :param points1: 在第一张图中的像素坐标
    :param points2: 在第二张图中的像素坐标
    :return: points4D：一串三维点，和上面的特征点数量相等
    '''
    #可参考帖子
    #https://blog.csdn.net/qq_38204686/article/details/115018686?ops_request_misc=&request_id=&biz_id=102&utm_term=%E4%B8%89%E8%A7%92%E6%B5%8B%E9%87%8F%E4%BB%A3%E7%A0%81%20python&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-6-115018686.142^v96^pc_search_result_base2&spm=1018.2226.3001.4187
    projMatr1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])  # 第一个相机外参
    projMatr2 = np.concatenate((R, t), axis=1)  # 第二个相机外参
    projMatr1 = np.matmul(camK, projMatr1)  # projection matrix投影矩阵，本质上就是二维点在三维上的投影变化
    projMatr2 = np.matmul(camK, projMatr2)
    points1_float = points1.astype(float)
    points2_float = points2.astype(float)
    points4D = cv2.triangulatePoints(projMatr1, projMatr2, points1_float.T, points2_float.T)
    points4D /= points4D[3]  # 出来是四维的，归一化
    points4D = points4D.T[:, 0:3]#返回三维点
    return points4D
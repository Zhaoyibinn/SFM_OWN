import numpy as np
import cv2

def rotation_vector2matrix(rotation_vector):
    '''
    将旋转向量转换为旋转矩阵
    :param rotation_vector: 旋转向量 numpy矩阵（3*1）
    :return: 对应的旋转矩阵 numpy矩阵（3*3）
    '''
    return cv2.Rodrigues(rotation_vector)[0]#罗德里格斯公式得到旋转矩阵

def Rt2SE3(R, t):
    '''
    从旋转矩阵R和平移向量t中拼接得到SE3
    :param R: 旋转矩阵 numpy （3*3）
    :param t: 平移向量 numpy （3*1,1*3,3）
    :return: SE3 特殊欧氏群 numpy（4*4）
    '''
    SE3 = np.identity(4)
    SE3[0:3, 0:3] = R
    SE3[0:3, -1]  = t.squeeze()
    return SE3

def SE32Rt(SE3):
    '''
    从SE3中分离得到旋转矩阵R和平移向量t
    :param SE3: 特殊欧氏群 numpy（4*4）
    :return: R: 旋转矩阵 numpy （3*3）
             t: 平移向量 numpy （3*1）
    '''
    R = SE3[0:3, 0:3]
    t = SE3[0:3, -1].reshape(-1, 1)
    return R, t
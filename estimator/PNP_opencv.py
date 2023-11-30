import numpy as np
import cv2

def EPNP(co_3dpoints, extra_pic_feature_xy, camK):
    '''
    调用opencv的EPNP求解PNP问题
    :param co_3dpoints:共视三维点 numpy矩阵（K*3）
    :param extra_pic_feature_xy:增量图片中的共视二维特征点 numpy矩阵（K*2）
    :param camK:相机内参
    :return: ok：
             rotation_vector: 旋转向量 numpy矩阵
             translation_vector: 平移向量 numpy矩阵
    '''
    ok, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(co_3dpoints.astype(np.float32), extra_pic_feature_xy.astype(np.float32), camK, np.zeros((4, 1)), flags=cv2.SOLVEPNP_EPNP)
    # ok, rotation_vector, translation_vector, _ = cv2.solvePnP(co_3dpoints.astype(np.float32),
    #                                                                 extra_pic_feature_xy.astype(np.float32), camK,
    #                                                                 np.zeros((4, 1)), flags=cv2.SOLVEPNP_EPNP)
    return ok, rotation_vector, translation_vector
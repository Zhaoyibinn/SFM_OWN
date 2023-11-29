import numpy as np
def pixel2cam(p,camK):
    '''
    像素坐标转相机坐标系下的归一化坐标
    :param p:像素坐标
    :param camK:相机内参
    :return: campoint：归一化坐标
    '''
    campoint = [(p[0]-camK[0,2])/camK[0,0],(p[1]-camK[1,2])/camK[1,1]]
    return campoint
def calerror(camK,list_kp1, list_kp2,points3d,R0,t0,R1,t1):
    '''
    用于计算两张图片之间三角化的重投影误差，参考高翔视觉里程计1，此处暂时不用，保留
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
    error_sum_1 = 0
    error_sum_2 = 0
    while i < len(list_kp1):
    #
        # print("正在计算点",end="")
        # print(i)


        pt1_cam = pixel2cam(list_kp1[i],camK);
        pt1_trans = np.dot(R0, points3d[i].T) + t0.T[0]
        pt1_trans = pt1_trans / pt1_trans[2]
        dis = ((pt1_cam[0] - pt1_trans[0]) ** 2 + (pt1_cam[1] - pt1_trans[1]) ** 2) ** 0.5
        error_sum_1 = error_sum_1 + dis

        # print( "point in the first camera frame: ",end="")
        # print( pt1_cam )
        # print("point projected from 3D ", end="")
        # print(pt1_cam_3d)
        # print(" ")
    #
    # // 第二个图
        pt2_cam = pixel2cam(list_kp2[i], camK);
        pt2_trans = np.dot(R1,points3d[i].T)+t1.T[0]
        pt2_trans=pt2_trans/pt2_trans[2]
        dis = ((pt2_cam[0] - pt2_trans[0]) ** 2 + (pt2_cam[1] - pt2_trans[1]) ** 2) ** 0.5
        error_sum_2 = error_sum_2 + dis

        # print("point in the second camera frame: ", end="")
        # print(pt2_cam)
        # print("point projected from 3D ", end="")
        # print(pt2_trans)
        # print(" ")
        # print(" ")
        # print("误差=", end="")


        i=i+1

    print("第一张图片的重新投影误差为：", end="")
    print(error_sum_1)
    print("第二张图片的重新投影误差为：", end="")
    print(error_sum_2)

    return 0
import numpy
import numpy as np
import SURF
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import networkx as nx

def create_co_see_pic(list_kp_1s,list_kp_2s,matchidxs,imgnum):
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


def sfm_1step(f_pic_idx ,co_pic_idx, s_pic_idx ,list_kp_1s,list_kp_2s,matchidxs,R01,t01,points3d01):
    list_kp1_12 = np.array(list_kp_1s[co_pic_idx][s_pic_idx])
    list_kp2_12 = np.array(list_kp_2s[co_pic_idx][s_pic_idx])
    list_kp1_01 = np.array(list_kp_1s[f_pic_idx][co_pic_idx])
    list_kp2_01 = np.array(list_kp_2s[f_pic_idx][co_pic_idx])
    # _, _, list_kp1_12, list_kp2_12, _,_ = first_proc_2pic(imgpaths, camK, imgnum, 1, 2)#这里只是用二维特征点信息

    co_feature_idx_idx = []
    for i in range(np.array(matchidxs[f_pic_idx][co_pic_idx]).shape[0]):
        for n in range(np.array(matchidxs[co_pic_idx][s_pic_idx]).shape[0]):
            if matchidxs[f_pic_idx][co_pic_idx][i][1] == matchidxs[co_pic_idx][s_pic_idx][n][0]:
                co_feature_idx_idx.append([i, n])
                # first_pic_feature_idx.append(matchidxs[f_pic_idx][co_pic_idx][i][0])
                # co_pic_feature_idx.append(matchidxs[f_pic_idx][co_pic_idx][i][1])
                # second_pic_feature_idx.append(matchidxs[co_pic_idx][s_pic_idx][i][1])
    co_3dpoints = []

    for i in range(co_feature_idx_idx.__len__()):
        co_3dpoints.append(points3d01[co_feature_idx_idx[i][0]])

    first_pic_feature_xy = []
    second_pic_feature_xy = []
    co_pic_feature_xy = []
    for i in range(co_feature_idx_idx.__len__()):
        second_pic_feature_xy.append(list_kp2_12[co_feature_idx_idx[i][1]])
        first_pic_feature_xy.append(list_kp1_01[co_feature_idx_idx[i][0]])
        co_pic_feature_xy.append(list_kp2_01[co_feature_idx_idx[i][0]])


    #
    # img1 = cv2.imread(imgpaths[f_pic_idx])#可视化这三张图片的共同特征点
    # img2 = cv2.imread(imgpaths[co_pic_idx])
    # img3 = cv2.imread(imgpaths[s_pic_idx])
    # look_3_co_pic(img1,img2,img3,first_pic_feature_xy,second_pic_feature_xy,co_pic_feature_xy,co_feature_idx_idx.__len__())
    #
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(co_3dpoints)
    # o3d.visualization.draw_geometries([pcd])



    # ok, rotation_vector, translation_vector= cv2.solvePnP(np.float32(co_3dpoints), np.float32(second_pic_feature_xy),
    #                                                        camK, np.zeros((4, 1)),flags=0)
    pnp_co_3dpoints = np.array([co_3dpoints], dtype=np.float32)
    pnp_second_pic_feature_xy = np.array([second_pic_feature_xy], dtype=np.float32)
    ok, rotation_vector, translation_vector= cv2.solvePnP(pnp_co_3dpoints,pnp_second_pic_feature_xy,
                                                           camK, np.zeros((4, 1)),flags=cv2.SOLVEPNP_EPNP)

    rotM = cv2.Rodrigues(rotation_vector)[0]
    transform_matrix02 = np.identity(4)  # 组合变换矩阵02
    transform_matrix02[0:3, 0:3] = rotM
    for i in range(0, 3, 1):
        transform_matrix02[i, 3] = translation_vector[i]
    transform_matrix01 = np.identity(4)  # 组合变换矩阵01
    transform_matrix01[0:3, 0:3] = R01
    for i in range(0, 3, 1):
        transform_matrix01[i, 3] = t01[i]
    transform_matrix12 = np.dot(np.linalg.inv(transform_matrix01), transform_matrix02)

    # 三角化
    pnp_rotM = transform_matrix12[:3, :3]
    pnp_t = transform_matrix12[:3, 3].reshape(-1, 1)
    points3d12_cam1 = triangulate(pnp_rotM, pnp_t, list_kp1_12, list_kp2_12)
    points3d12 = (np.dot(np.linalg.inv(R01), (points3d12_cam1.T - t01))).T
    return points3d12,translation_vector,rotM

def look_3_co_pic(img1,img2,img3,first_pic_feature_xy,second_pic_feature_xy,co_pic_feature_xy,co_feature_num):#可视化共视图
    for i in range(co_feature_num):
        img1 = cv2.circle(img1,(first_pic_feature_xy[i][0],first_pic_feature_xy[i][1]),14,(255, 0, 0),-1)
        img2 = cv2.circle(img2, (co_pic_feature_xy[i][0],co_pic_feature_xy[i][1]),14,(255, 0, 0),-1)
        img3 = cv2.circle(img3, (second_pic_feature_xy[i][0],second_pic_feature_xy[i][1]),14,(255, 0, 0),-1)

    plt.subplot(131)
    plt.imshow(img1)
    plt.subplot(132)
    plt.imshow(img2)
    plt.subplot(133)
    plt.imshow(img3)
    plt.show()

    return 0



def get_color(depth):
    up_th = 7
    low_th = 1
    th_range = up_th - low_th;
    if (depth > up_th):
        depth = up_th;
    if (depth < low_th):
        depth = low_th;
    return (255 * (depth-low_th) / th_range, 0, 255 * (1 - (depth-low_th) / th_range));

def pixel2cam(p,camK):
    campoint = [(p[0]-camK[0,2])/camK[0,0],(p[1]-camK[1,2])/camK[1,1]]
    return campoint


def calerror(camK,list_kp1, list_kp2,points3d,R,t):
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
def triangulate(R, t ,points1,points2):#https://blog.csdn.net/qq_38204686/article/details/115018686?ops_request_misc=&request_id=&biz_id=102&utm_term=%E4%B8%89%E8%A7%92%E6%B5%8B%E9%87%8F%E4%BB%A3%E7%A0%81%20python&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-6-115018686.142^v96^pc_search_result_base2&spm=1018.2226.3001.4187
    projMatr1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])  # 第一个相机参数
    projMatr2 = np.concatenate((R, t), axis=1)  # 第二个相机参数
    projMatr1 = np.matmul(camK, projMatr1)  # 相机内参 相机外参
    projMatr2 = np.matmul(camK, projMatr2)  #
    points1_float = points1.astype(float)
    points2_float = points2.astype(float)
    points4D = cv2.triangulatePoints(projMatr1, projMatr2, points1_float.T, points2_float.T)
    points4D /= points4D[3]  # 归一化
    points4D = points4D.T[:, 0:3]
    return points4D

def E2RT(E,list_kp1, list_kp2): #本质矩阵转RT

    retval, R, t, mask = cv2.recoverPose(E,list_kp1, list_kp2, camK)
    return R,t

def first_proc_2pic(imgpaths,camK,imgnum,picidx1,picidx2):
    list_kp_1s, list_kp_2s,imgcolors ,matchidxs= SURF.surf(imgpaths,imgnum)
    list_kp1 = numpy.array(list_kp_1s[picidx1][picidx2])#这里就直接是图[picidx1]和图[picidx2]的像素坐标点
    list_kp2 = numpy.array(list_kp_2s[picidx1][picidx2])
    good_F, status = cv2.findFundamentalMat(list_kp1, list_kp2, method=cv2.FM_RANSAC, ransacReprojThreshold=3,
                                            confidence=0.99)

    print("基础矩阵F=", end="")
    print(good_F)
    E = np.dot(np.dot(np.transpose(camK), good_F), camK)
    print("本质矩阵E=", end="")
    print(E)

    R, t = E2RT(E, list_kp1, list_kp2)  # 计算得到R，t
    print("计算得到R=", end="")
    print(R)
    print("计算得到t=", end="")
    print(t)
    # test R t
    # R=[[ 0.97220802, 0.12501667 ,-0.19794913],
    #  [-0.12103169  ,0.99212784  ,0.03214906],
    #  [ 0.20040998 ,-0.00729785  ,0.97968459]]
    points3d = triangulate(R, t, list_kp1, list_kp2)

    # calerror(camK, list_kp1, list_kp2, points3d, R, t) #计算重投影误差

    return R,t,list_kp1, list_kp2,points3d,matchidxs

def main(imgpaths,camK,imgnum):
    R01,t01,list_kp1_01, list_kp2_01,points3d01,matchidxs= first_proc_2pic(imgpaths,camK,imgnum,0,1)
    list_kp_1s, list_kp_2s, imgcolors, matchidxs = SURF.surf(imgpaths, imgnum)
    create_co_see_pic(list_kp_1s, list_kp_2s, matchidxs,imgnum)

    points3d12 , t02 , R02 = sfm_1step(0,1,2,list_kp_1s,list_kp_2s,matchidxs,R01,t01,points3d01)
    points3d23, t03, R03 = sfm_1step(1, 2, 3, list_kp_1s, list_kp_2s, matchidxs, R02, t02, points3d12)






    pcd01 = o3d.geometry.PointCloud()
    pcd01.points = o3d.utility.Vector3dVector(points3d01)
    pcd01.paint_uniform_color([0, 0, 1])

    pcd12 = o3d.geometry.PointCloud()  # 展示三维点
    pcd12.points = o3d.utility.Vector3dVector(points3d12)
    pcd12.paint_uniform_color([1, 0, 0])

    pcd23 = o3d.geometry.PointCloud()
    pcd23.points = o3d.utility.Vector3dVector(points3d23)
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
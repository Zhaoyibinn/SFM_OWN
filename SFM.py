import numpy
import numpy as np
import SURF
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import networkx as nx

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


def sfm_1step(f_pic_idx ,co_pic_idx, s_pic_idx ,list_kp_1s,list_kp_2s,matchidxs,R01,t01,points3d01):
    '''
    增量式构建，主要是PNP+三角化
    :param f_pic_idx:第一张图片的序号（此处指全新的图片之前的第二张）
    :param co_pic_idx:第二张图片的序号（此处指全新的图片之前的第一张）
    :param s_pic_idx:第三张图片的序号（全新的图片）
    :param list_kp_1s:特征匹配数组中第一张图片的特征点像素坐标，N*N个格子（N为图片数量），每个格子里面存着许多对特征点
    :param list_kp_2s:特征匹配数组中第二张图片的特征点像素坐标
    :param matchidxs:特征匹配idx数组，N*N个格子，每个格子里面存着所有特征点对所对应的图片特征点idx
    :param R01:第二张图片的相机旋转矩阵（也就是第二张图片相对于第一张图片的相机位姿，因为全部算法是以第一张图片的相机位姿为原点的）
    :param t01:第二张图片的相机平移向量
    :param points3d01:第一张和第二张图片构造的点云
    :return:    points3d12：第二张图片和第三张图片构造的三维点；
                translation_vector：第三张图片相机的平移向量；
                rotM：第三张图片相机的旋转矩阵
    '''
    list_kp1_12 = np.array(list_kp_1s[co_pic_idx][s_pic_idx])#提取出来两对图片各自的特征点像素坐标，共计四组
    list_kp2_12 = np.array(list_kp_2s[co_pic_idx][s_pic_idx])
    list_kp1_01 = np.array(list_kp_1s[f_pic_idx][co_pic_idx])
    list_kp2_01 = np.array(list_kp_2s[f_pic_idx][co_pic_idx])

    # 计算三张图片的共同特征，通过比对matchidx
    co_feature_idx_idx = []#第一张图和第三张图的共同特征点序号数组（这里用的不是特征匹配的idx，直接用的是匹配完后的数组序号，即第n对匹配点）
    for i in range(np.array(matchidxs[f_pic_idx][co_pic_idx]).shape[0]):
        for n in range(np.array(matchidxs[co_pic_idx][s_pic_idx]).shape[0]):
            if matchidxs[f_pic_idx][co_pic_idx][i][1] == matchidxs[co_pic_idx][s_pic_idx][n][0]:
                co_feature_idx_idx.append([i, n])
                # first_pic_feature_idx.append(matchidxs[f_pic_idx][co_pic_idx][i][0])
                # co_pic_feature_idx.append(matchidxs[f_pic_idx][co_pic_idx][i][1])
                # second_pic_feature_idx.append(matchidxs[co_pic_idx][s_pic_idx][i][1])

    # 通过上面的共同特征点找到这些特征点的三维坐标（在第一张和第二张图三角化的点云中找）
    co_3dpoints = []
    for i in range(co_feature_idx_idx.__len__()):
        co_3dpoints.append(points3d01[co_feature_idx_idx[i][0]])

    # 通过上面的共同特征点找到这些特征点的二维像素坐标
    first_pic_feature_xy = []#第一张图片的共视特征点像素坐标
    second_pic_feature_xy = []#第三张图片的共视特征点像素坐标
    co_pic_feature_xy = []#第二张图片的共视特征点像素坐标
    for i in range(co_feature_idx_idx.__len__()):
        second_pic_feature_xy.append(list_kp2_12[co_feature_idx_idx[i][1]])
        first_pic_feature_xy.append(list_kp1_01[co_feature_idx_idx[i][0]])
        co_pic_feature_xy.append(list_kp2_01[co_feature_idx_idx[i][0]])


    #三张图片共视点可视化
    #这里二维可视化和三维可视化好像会冲突，建议只看一个，两个都要看就只能看一次，再次进这个函数就会报错（不知道为什么）
    img1 = cv2.imread(imgpaths[f_pic_idx])#二维可视化
    img2 = cv2.imread(imgpaths[co_pic_idx])
    img3 = cv2.imread(imgpaths[s_pic_idx])
    look_3_co_pic(img1,img2,img3,first_pic_feature_xy,second_pic_feature_xy,co_pic_feature_xy,co_feature_idx_idx.__len__())
    #
    # pcd = o3d.geometry.PointCloud()#三维可视化
    # pcd.points = o3d.utility.Vector3dVector(co_3dpoints)
    # o3d.visualization.draw_geometries([pcd])




    pnp_co_3dpoints = np.array([co_3dpoints], dtype=np.float32)#此处需要套一层，反正也是opencv的数据规矩，不然不能用EPNP
    pnp_second_pic_feature_xy = np.array([second_pic_feature_xy], dtype=np.float32)
    ok, rotation_vector, translation_vector,_= cv2.solvePnPRansac(pnp_co_3dpoints,pnp_second_pic_feature_xy,#这里用的Ransac，不过实际实验下来用不用没啥大区别
                                                           camK, np.zeros((4, 1)),flags=cv2.SOLVEPNP_EPNP)
    # ok, rotation_vector, translation_vector= cv2.solvePnP(np.float32(co_3dpoints), np.float32(second_pic_feature_xy),
    #                                                        camK, np.zeros((4, 1)),flags=0)

    rotM = cv2.Rodrigues(rotation_vector)[0]#罗德里格斯公式得到旋转矩阵
    transform_matrix02 = np.identity(4)     #组合变换矩阵02（第一张图和第三张图）
    transform_matrix02[0:3, 0:3] = rotM
    for i in range(0, 3, 1):
        transform_matrix02[i, 3] = translation_vector[i]
    transform_matrix01 = np.identity(4)  # 组合变换矩阵01（第一张图和第二张图）
    transform_matrix01[0:3, 0:3] = R01
    for i in range(0, 3, 1):
        transform_matrix01[i, 3] = t01[i]
    transform_matrix12 = np.dot(np.linalg.inv(transform_matrix01), transform_matrix02)#计算变换矩阵12（第二张图和第三张图）

    # 三角化
    pnp_rotM = transform_matrix12[:3, :3]
    pnp_t = transform_matrix12[:3, 3].reshape(-1, 1)
    points3d12_cam1 = triangulate(pnp_rotM, pnp_t, list_kp1_12, list_kp2_12)

    points3d12 = (np.dot(np.linalg.inv(R01), (points3d12_cam1.T - t01))).T#还原到世界坐标下（原本是在第二张图的摄像机坐标系下的）
    return points3d12,translation_vector,rotM

def look_3_co_pic(img1,img2,img3,first_pic_feature_xy,second_pic_feature_xy,co_pic_feature_xy,co_feature_num):
    '''
    共视特征点二维可视化
    :param img1: 第一张图片
    :param img2: 第二张图片
    :param img3: 第三张图片
    :param first_pic_feature_xy: 第一张图片的共视特征点像素坐标
    :param second_pic_feature_xy: 第三张图片的共视特征点像素坐标
    :param co_pic_feature_xy: 第二张图片的共视特征点像素坐标
    :param co_feature_num: 共视特征点数量
    :return: None
    '''
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

def triangulate(R, t ,points1,points2):
    '''
    三角化
    :param R:
    :param t:
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



def first_proc_2pic(imgpaths,camK,imgnum,picidx1,picidx2):
    '''
    最开始的两张图片处理，用对极几何求解R，t，并三角化计算得到三维点作为初始点云
    :param imgpaths: 全部图片路径数组
    :param camK: 相机内参
    :param imgnum: 图片数量
    :param picidx1: 需要处理的图片1
    :param picidx2: 需要处理的图片1
    :return:    R：旋转矩阵；
                t：平移向量；
                list_kp1：匹配好的特征点在第1张图中的像素坐标；
                list_kp2：匹配好的特征点在第2张图中的像素坐标；
                points3d：三维坐标点；
                matchidxs：匹配的对应idx，即前一张图片的特征点idx和后一张图片的特征点idx，为N*2数组
    '''
    list_kp_1s, list_kp_2s,imgcolors ,matchidxs= SURF.surf(imgpaths,imgnum)
    list_kp1 = numpy.array(list_kp_1s[picidx1][picidx2])#这里就直接是图[picidx1]和图[picidx2]的像素坐标点
    list_kp2 = numpy.array(list_kp_2s[picidx1][picidx2])
    good_F, status = cv2.findFundamentalMat(list_kp1, list_kp2, method=cv2.FM_RANSAC, ransacReprojThreshold=3,
                                            confidence=0.99)#使用RANSAC方法计算基本矩阵，函数参考
                                                            #https://blog.csdn.net/bb_sy_w/article/details/121082013?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170108654916800215081297%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170108654916800215081297&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-5-121082013-null-null.142^v96^pc_search_result_base2&utm_term=cv2.findFundamentalMat&spm=1018.2226.3001.4187
    print("前两张图片的基础矩阵F=", end="")
    print(good_F)
    E = np.dot(np.dot(np.transpose(camK), good_F), camK)#计算本质矩阵，就是(K.T)*F*K
    print("前两张图片的本质矩阵E=", end="")
    print(E)
    retval, R, t, mask = cv2.recoverPose(E, list_kp1, list_kp2, camK)  # 计算得到R，t
    print("计算得到前两张图片的R=", end="")
    print(R)
    print("计算得到前两张图片的t=", end="")
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
    #此处暂时进行了四张图片，三张点云的构建
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
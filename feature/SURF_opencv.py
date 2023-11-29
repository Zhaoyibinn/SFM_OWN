import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from feature import my_feature_class
class SURF_cv2(my_feature_class.feature):
    '''
        基于opencv实现的SURF特征点提取、描述、匹配
        :param imgpaths: 全部图片路径数组
        :param imgnum: 图片数量
    '''

    def __init__(self, imgpaths:list, imgnum:int):
        # 调用父类的初始化方法
        super().__init__(imgpaths, imgnum)
    def surf_feature_extraction(self):
        '''
        :return:list_kp_1s：特征匹配数组中第一张图片的特征点像素坐标，N*N个格子（N为图片数量），每个格子里面存着许多对特征点;
                list_kp_2s：特征匹配数组中第二张图片的特征点像素坐标;
                imgcolors：彩色图片;
                matchidxs：特征匹配idx数组，N*N个格子，每个格子里面存着所有特征点对所对应的图片特征点idx
        '''
        imgs = []  # 灰度图
        for i in range(self.imgnum_):
            imgs.append(cv2.imread(self.imgpaths_[i], 0))
        height, width = imgs[0].shape

        k = 1  # 缩放系数，主要是防止图片太大不方便观察，这里图片合适就直接取1不缩放
        for i in range(self.imgnum_):
            imgs[i] = cv2.resize(imgs[i], (int(width / k), int(height / k)), interpolation=cv2.INTER_LINEAR)
        # ------------ 提取特征与相应的描述子 ------------
        key_querys = []  # 关键点数组
        desc_querys = []  # 描述子数组
        for i in range(self.imgnum_):
            key_query, desc_query = self.__surf_detect(imgs[i])
            key_querys.append(key_query)
            desc_querys.append(desc_query)
        # ------------ 进行特征匹配 ------------
        goodmatches = []  # 好的匹配特征数组
        matches = []  # 匹配特征数组，没用到
        # 第i张图片和后面的第i+1,i+2,...,imgnum-1张图片进行特征匹配
        for i in range(self.imgnum_-1):
            n = i + 1
            goodmatch = []
            match = []
            while n < self.imgnum_:
                onegoodmatch, onematch = self.__surf_match(key_querys[i], desc_querys[i], key_querys[n], desc_querys[n])
                goodmatch.append(onegoodmatch)
                match.append(onematch)
                n = n + 1
            goodmatches.append(goodmatch)
            matches.append(match)
        # ------------ 获取特征匹配对应点的像素坐标值与索引 ------------
        matchidxs = []  # 特征匹配idx数组，N*N个格子，每个格子里面存着所有特征点对所对应的图片特征点idx
        list_kp_1s = []  # 特征匹配数组中第一张图片的特征点像素坐标，N*N个格子（N为图片数量），每个格子里面存着许多对特征点;
        list_kp_2s = []  # 特征匹配数组中第二张图片的特征点像素坐标
        # list_kp_1s[i][j]代表了第i张图片和第j张图片进行特征匹配时，图片i的特征点像素坐标
        # list_kp_2s[i][j]代表了第i张图片和第j张图片进行特征匹配时，图片j的特征点像素坐标
        for i in range(self.imgnum_-1):
            n = i + 1
            matchidx = []
            list_kp_1 = []
            list_kp_2 = []
            for j in range(len(goodmatches[i])):
                one_list_kp_1, one_list_kp_2, one_matchidx = self.__get_coordinates_from_matches(goodmatches[i][j],
                                                                                          key_querys[i],
                                                                                          key_querys[n])
                matchidx.append(one_matchidx)
                list_kp_1.append(one_list_kp_1)
                list_kp_2.append(one_list_kp_2)
                n = n + 1
            matchidxs.append(matchidx)
            list_kp_1s.append(list_kp_1)
            list_kp_2s.append(list_kp_2)
        # return list_kp_1s, list_kp_2s, imgcolors, matchidxs
        self.list_kp_1s_ = list_kp_1s
        self.list_kp_2s_ = list_kp_2s
        self.matchidxs_  = matchidxs
        return list_kp_1s, list_kp_2s, matchidxs

    def get_feature(self, idx_pic1, idx_pic2):
        '''
        获取idx_pic1和idx_pic2之间进行特征匹配后得到的特征点的像素坐标。
        其中idx_pic1必须小于idx_pic2，不然则抛出异常
        :param idx_pic1:要获取特征点像素坐标的图片1的索引
        :param idx_pic2:要获取特征点像素坐标的图片2的索引
        :return:图片1中的特征点的像素坐标：numpy矩阵（K*2）
                图片2中的特征点的像素坐标：numpy矩阵（K*2）
                图片1和图片2之间匹配的特征点在各自特征点中的索引对：numpy矩阵（K*2）
        '''
        # 因为列表中缺少占位符，所以原来的图片索引要要减去1再减去第一张图片的索引。
        if idx_pic1 < idx_pic2:
            return np.array(self.list_kp_1s_[idx_pic1][idx_pic2-idx_pic1-1]), np.array(self.list_kp_2s_[idx_pic1][idx_pic2-idx_pic1-1]), np.array(self.matchidxs_[idx_pic1][idx_pic2 - idx_pic1 - 1])
        else:
            raise ValueError("idx_pic1必须小于idx_pic2")

    def get_co_feature(self, idx_pic1, idx_pic2, idx_pic3, Visual=False):
        '''
        给定3个图片的增序索引，查找在三张图片中都出现的特帧点，返回特征点（1,2之间）的顺序索引
        :param idx_pic1: 要获取特征点像素坐标的图片1的索引
        :param idx_pic2: 要获取特征点像素坐标的图片2的索引
        :param idx_pic3: 要获取特征点像素坐标的图片3的索引
        :param Visual:   是否进行共视特征点的可视化，默认不进行可视化
        :return: co_idx_12 （共视）特征点（1,2之间）的顺序索引
                 co_feature12_pic1_xy （共视）特征点在图片1中的像素坐标
                 co_feature12_pic2_xy （共视）特征点在图片2中的像素坐标
                 co_feature23_pic3_xy （共视）特征点在图片3中的像素坐标
        '''
        if idx_pic1 < idx_pic2 and idx_pic2 < idx_pic3:
            # 图1和图2之间进行匹配，得到的特征点像素坐标
            list_kp1_12 = np.array(self.list_kp_1s_[idx_pic1][idx_pic2 - idx_pic1 - 1])
            list_kp2_12 = np.array(self.list_kp_2s_[idx_pic1][idx_pic2 - idx_pic1 - 1])
            # 图2和图3之间进行匹配，得到的特征点像素坐标
            list_kp2_23 = np.array(self.list_kp_1s_[idx_pic2][idx_pic3 - idx_pic2 - 1])
            list_kp3_23 = np.array(self.list_kp_2s_[idx_pic2][idx_pic3 - idx_pic2 - 1])
            # 图片1和图片2之间的可以进行匹配的特征点索引对
            matches_between_12 = np.array(self.matchidxs_[idx_pic1][idx_pic2 - idx_pic1 - 1])
            # 图片2和图片3之间的可以进行匹配的特征点索引对
            matches_between_23 = np.array(self.matchidxs_[idx_pic2][idx_pic3 - idx_pic2 - 1])
            # 利用广播机制，构建布尔值矩阵，寻找图片2中，和图片1、图片3都构成匹配点对的特征点的索引
            bool_mat = matches_between_12[:, -1].reshape(-1, 1) == matches_between_23[:, 0].reshape(1, -1)
            # idx_mat是一个矩阵（2*共同特征点数），每一列包含形如array([a, b])的numpy矩阵。
            # a,b分别代表了共同特征点的索引在matches_between_12和matches_between_23中所处的行数。
            idx_mat = np.asarray(np.where(bool_mat))
            co_idx_12 = idx_mat[0, :]
            co_idx_23 = idx_mat[1, :]
            # 通过上面的共视特征点的索引找到这些特征点的二维像素坐标
            co_feature12_pic1_xy = list_kp1_12[co_idx_12]
            co_feature12_pic2_xy = list_kp2_12[co_idx_12]
            co_feature23_pic3_xy = list_kp3_23[co_idx_23]
            # 三张图片共视点可视化
            # 这里二维可视化和三维可视化好像会冲突，建议只看一个，两个都要看就只能看一次，再次进这个函数就会报错（不知道为什么）
            img1 = cv2.imread(self.imgpaths_[idx_pic1])  # 二维可视化
            img2 = cv2.imread(self.imgpaths_[idx_pic2])
            img3 = cv2.imread(self.imgpaths_[idx_pic3])
            if Visual:
                self.__look_3_co_pic(img1, img2, img3, co_feature12_pic1_xy, co_feature12_pic2_xy, co_feature23_pic3_xy, idx_mat.shape[1])
        else:
            raise ValueError("输入必须增序排列")
        return co_idx_12, co_feature12_pic1_xy, co_feature12_pic2_xy, co_feature23_pic3_xy
    def __surf_detect(self, img):
        '''
        SURF特征点检测
        :param img: 一张灰度图
        :return:    key_query：关键点，存储了坐标；
                    desc_query：描述符，存储了特征
        '''
        surf = cv2.xfeatures2d_SURF.create(2000)
        # 找到关键点和描述符
        key_query, desc_query = surf.detectAndCompute(img, None)
        return key_query, desc_query

    def __surf_match(self, key_query1, desc_query1, key_query2, desc_query2):
        '''
        SURF特征匹配
        :param key_query1: 第一张图片的关键点
        :param desc_query1: 第一张图片的描述符
        :param key_query2: 第二张图片的关键点
        :param desc_query2: 第二张图片的描述符
        :return:    goodmatches：两张图较好的匹配，用DMatch封装，后续用这个；
                    matches：两张图全部的匹配
        '''
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_query1, desc_query2, k=2)  # 一对多匹配

        goodmatches = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:  # 最优点的距离（可理解为残差）是次优点的一半以下可以被认为是好的点
                goodmatches.append([m])

        return goodmatches, matches

    def __get_coordinates_from_matches(self, matches: list, kp1: list, kp2: list):
        '''
        将Dmatch数据解包，直接得到像素坐标和对应匹配idx
        :param matches: 两张图片的match
        :param kp1: 第一张图片的关键点
        :param kp2: 第二张图片的关键点
        :return:    list_kp1：第一张图片的特征点像素坐标；
                    list_kp2：第二张图片的特征点像素坐标；
                    matchidx：两张图片特征匹配的对应特征点idx
        '''
        # Initialize lists
        list_kp1 = []
        list_kp2 = []
        matchidx = []
        # For each match...
        for mat in matches:
            # Get the matching keypoints for each of the images
            img1_idx = mat[0].queryIdx
            img2_idx = mat[0].trainIdx
            matchidx.append([img1_idx, img2_idx])
            # x - columns
            # y - rows
            # Get the coordinates
            (x1, y1) = (int(kp1[img1_idx].pt[0]), int(kp1[img1_idx].pt[1]))
            (x2, y2) = (int(kp2[img2_idx].pt[0]), int(kp2[img2_idx].pt[1]))
            # Append to each list
            list_kp1.append((x1, y1))
            list_kp2.append((x2, y2))

        return list_kp1, list_kp2, matchidx

    def __look_3_co_pic(self, img1, img2, img3,  co_feature12_pic1_xy, co_feature12_pic2_xy, co_feature23_pic3_xy, co_feature_num):
        '''
        共视特征点二维可视化
        :param img1: 第一张图片
        :param img2: 第二张图片
        :param img3: 第三张图片
        :param co_feature12_pic1_xy: 第一张图片的共视特征点像素坐标
        :param co_feature12_pic2_xy: 第三张图片的共视特征点像素坐标
        :param co_feature23_pic3_xy: 第二张图片的共视特征点像素坐标
        :param co_feature_num: 共视特征点数量
        :return: None
        '''
        for i in range(co_feature_num):
            img1 = cv2.circle(img1, (co_feature12_pic1_xy[i][0], co_feature12_pic1_xy[i][1]), 14, (255, 0, 0), -1)
            img2 = cv2.circle(img2, (co_feature12_pic2_xy[i][0], co_feature12_pic2_xy[i][1]), 14, (255, 0, 0), -1)
            img3 = cv2.circle(img3, (co_feature23_pic3_xy[i][0], co_feature23_pic3_xy[i][1]), 14, (255, 0, 0), -1)
        # img1 = cv2.circle(img1, (co_feature12_pic1_xy[:, 0], co_feature12_pic1_xy[:, 1]), 14, (255, 0, 0), -1)
        # img2 = cv2.circle(img2, (co_feature12_pic2_xy[:, 0], co_feature12_pic2_xy[:, 1]), 14, (255, 0, 0), -1)
        # img3 = cv2.circle(img3, (co_feature23_pic3_xy[:, 0], co_feature23_pic3_xy[:, 0]), 14, (255, 0, 0), -1)

        plt.subplot(131)
        plt.imshow(img1)
        plt.subplot(132)
        plt.imshow(img2)
        plt.subplot(133)
        plt.imshow(img3)
        plt.show()

        return 0
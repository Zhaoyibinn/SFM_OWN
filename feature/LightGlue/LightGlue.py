from feature import my_feature_class
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import torch

class lightglue(my_feature_class.feature):
    '''
        LightGlue特征点提取、描述、匹配
        :param imgpaths: 全部图片路径数组
        :param imgnum: 图片数量
    '''
    def __init__(self, imgpaths: list, imgnum: int):
        # 调用父类的初始化方法
        self.extractor = SuperPoint(max_num_keypoints=512).eval().cuda()
        self.matcher = LightGlue(features='superpoint').eval().cuda()
        super().__init__(imgpaths, imgnum)


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

        # image0 = load_image(self.imgpaths_[idx_pic1]).cuda()
        # image1 = load_image(self.imgpaths_[idx_pic2]).cuda()
        # img0 = cv2.imread(self.imgpaths_[idx_pic1])
        # img1 = cv2.imread(self.imgpaths_[idx_pic2])
        # feats0 = self.extractor.extract(image0)
        # feats1 = self.extractor.extract(image1)
        # matches01 = self.matcher({'image0': feats0, 'image1': feats1})
        #
        # points0 = feats0['keypoints'][0]  # coordinates in image #0, shape (K,2)
        # points1 = feats1['keypoints'][0]  # coordinates in image #1, shape (K,2)
        # list_kp_1s = np.array(points0.cpu()).astype(int)
        # list_kp_2s = np.array(points1.cpu()).astype(int)
        # matchidxs = np.array(matches01['matches'][0].cpu())

        img0 = cv2.imread(self.imgpaths_[idx_pic1])
        img1 = cv2.imread(self.imgpaths_[idx_pic2])
        points0 = self.list_kp_1s_[idx_pic1][idx_pic2-idx_pic1-1]
        points1 = self.list_kp_2s_[idx_pic1][idx_pic2-idx_pic1-1]

        # self.match_visual(img0, img1, points0, points1)

        list_kp_1s = np.array(points0)
        list_kp_2s = np.array(points1)
        matchidxs = np.array(self.matchidxs_[idx_pic1][idx_pic2-idx_pic1-1])

        return list_kp_1s, list_kp_2s, matchidxs

    def lightglue_extract_feature(self):
        imgs = []  # 灰度图
        for i in range(self.imgnum_):
            imgs.append(cv2.imread(self.imgpaths_[i], 0))
        height, width = imgs[0].shape

        k = 1  # 缩放系数，主要是防止图片太大不方便观察，这里图片合适就直接取1不缩放
        for i in range(self.imgnum_):
            imgs[i] = cv2.resize(imgs[i], (int(width / k), int(height / k)), interpolation=cv2.INTER_LINEAR)
        # ------------ 提取特征与相应的描述子 ------------
        key_querys = []  # 关键点数组
        feats_querys = []  # 特征提取全部信息数组
        for i in range(self.imgnum_):
            image0 = load_image(self.imgpaths_[i]).cuda()
            feats0 = self.extractor.extract(image0)
            points0 = feats0['keypoints'][0]  # coordinates in image #0, shape (K,2)
            key_querys.append(np.array(points0.cpu()).astype(int))
            feats_querys.append(feats0)


        # ------------ 进行特征匹配 ------------
        goodmatches = []  # 好的匹配特征数组
        # 第i张图片和后面的第i+1,i+2,...,imgnum-1张图片进行特征匹配
        for i in range(self.imgnum_ - 1):
            n = i + 1
            goodmatch = []
            while n < self.imgnum_:


                feats0 = feats_querys[i]
                feats1 = feats_querys[n]
                matches01 = self.matcher({'image0': feats0, 'image1': feats1})
                matchidxs = np.array(matches01['matches'][0].cpu())
                goodmatch.append(matchidxs)
                n = n + 1
            goodmatches.append(goodmatch)

        # ------------ 获取特征匹配对应点的像素坐标值与索引 ------------
        matchidxs = []  # 特征匹配idx数组，N*N个格子，每个格子里面存着所有特征点对所对应的图片特征点idx
        list_kp_1s = []  # 特征匹配数组中第一张图片的特征点像素坐标，N*N个格子（N为图片数量），每个格子里面存着许多对特征点;
        list_kp_2s = []  # 特征匹配数组中第二张图片的特征点像素坐标
        # list_kp_1s[i][j]代表了第i张图片和第j张图片进行特征匹配时，图片i的特征点像素坐标
        # list_kp_2s[i][j]代表了第i张图片和第j张图片进行特征匹配时，图片j的特征点像素坐标
        for i in range(self.imgnum_ - 1):
            n = i + 1
            matchidx = []
            list_kp_1 = []
            list_kp_2 = []
            for j in range(len(goodmatches[i])):
                one_matchidx = goodmatches[i][j]
                one_list_kp_1, one_list_kp_2 = self.__get_coordinates_from_matches(one_matchidx,key_querys[i], key_querys[n])
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
        self.matchidxs_ = matchidxs
        return list_kp_1s, list_kp_2s, matchidxs

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

    def match_visual(self,img0, img1, points0, points1):
        newWidth = img0.shape[1] + img1.shape[1]
        newHeight = img0.shape[0]
        allpic = np.zeros((newHeight, newWidth, 3), np.uint8)
        allpic[:, 0:img0.shape[1]] = img0
        allpic[:, img0.shape[1]:newWidth] = img1
        for i in range(len(points0)):
            allpic = cv2.circle(allpic, (points0[i][0], points0[i][1]), 3, (255, 0, 0), -1)
            allpic = cv2.circle(allpic, (points1[i][0] + img0.shape[1], points1[i][1]), 3,
                                (255, 0, 0), -1)
        # plt.imshow(allpic)
        # plt.show()

        for i in range(len(points0)):
            allpic = cv2.line(allpic, (points0[i][0], points0[i][1]),
                              (points1[i][0] + img0.shape[1], points1[i][1]),
                              (255, 0, 0), 1)
        plt.imshow(allpic)
        plt.show()
        return 0

    def __get_coordinates_from_matches(self,matches,points0, points1):
        one_list_kp_1 = []
        one_list_kp_2 = []
        for i in range(len(matches)):
            one_list_kp_1.append([points0[matches[i][0]][0], points0[matches[i][0]][1]])
            one_list_kp_2.append([points1[matches[i][1]][0], points1[matches[i][1]][1]])

        return one_list_kp_1, one_list_kp_2

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


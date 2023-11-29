import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class feature(ABC):
    def __init__(self, imgpaths:list, imgnum:int):
        '''
            基类实现特征点提取、描述、匹配，其他提取特征的类都从它这里继承
            :param imgpaths: 全部图片路径数组
            :param imgnum: 图片数量
        '''
        self.imgpaths_ = imgpaths
        self.imgnum_ = imgnum

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass
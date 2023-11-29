import time

import cv2
import numpy as np
import matplotlib.pyplot as plt



def get_coordinates_from_matches(matches: list, kp1: list, kp2: list):
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
    matchidx= []
    # For each match...
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat[0].queryIdx
        img2_idx = mat[0].trainIdx
        matchidx.append([img1_idx,img2_idx])
        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = (int(kp1[img1_idx].pt[0]),int(kp1[img1_idx].pt[1]))
        (x2, y2) = (int(kp2[img2_idx].pt[0]),int(kp2[img2_idx].pt[1]))
        # Append to each list
        list_kp1.append((x1, y1))
        list_kp2.append((x2, y2))

    return list_kp1, list_kp2,matchidx


def surf_detect(img):
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

def surf_match(key_query1, desc_query1,key_query2, desc_query2):
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
    matches = bf.knnMatch(desc_query1, desc_query2, k=2)#一对多匹配

    goodmatches=[]
    for m, n in matches:
        if m.distance < 0.5 * n.distance:#最优点的距离（可理解为残差）是次优点的一半以下可以被认为是好的点
            goodmatches.append([m])

    return goodmatches,matches

def surf(imgpaths,imgnum):
    '''
    SURF特征提取，特征匹配
    :param imgpaths: 全部图片路径数组
    :param imgnum: 图片数量
    :return:    list_kp_1s：特征匹配数组中第一张图片的特征点像素坐标，N*N个格子（N为图片数量），每个格子里面存着许多对特征点;
                list_kp_2s：特征匹配数组中第二张图片的特征点像素坐标;
                imgcolors：彩色图片;
                matchidxs：特征匹配idx数组，N*N个格子，每个格子里面存着所有特征点对所对应的图片特征点idx
    '''
    imgcolors = []#彩色图
    for i in range(imgnum):
        imgcolors.append(cv2.imread(imgpaths[i]))

    imgs = []#灰度图
    for i in range(imgnum):
        imgs.append(cv2.imread(imgpaths[i],0))
    height, width = imgs[0].shape

    k=1 #缩放系数，主要是防止图片太大不方便观察，这里图片合适就直接取1不缩放
    for i in range(imgnum):
        imgs[i]= cv2.resize(imgs[i], (int(width/k), int(height/k)), interpolation=cv2.INTER_LINEAR)

    key_querys = []#关键点数组
    desc_querys = []#描述子数组
    for i in range(imgnum):
        key_query, desc_query = surf_detect(imgs[i])
        key_querys.append(key_query)
        desc_querys.append(desc_query)

    # detectimg = cv2.drawKeypoints(imgs[2], key_querys[2], imgs[2])#可视化特征提取，需要手动修改序号
    # cv2.imshow('sp1',detectimg)
    # cv2.waitKey(0)
    # detectimg = cv2.drawKeypoints(imgs[3], key_querys[3], imgs[3])#可视化特征提取，需要手动修改序号
    # cv2.imshow('sp2',detectimg)
    # cv2.waitKey(0)

    goodmatches = []#好的匹配特征数组
    matches = []#匹配特征数组，没用到
    for i in range (imgnum):
        n = 0
        goodmatch=[]
        match=[]
        while n < imgnum:
            onegoodmatch, onematch = surf_match(key_querys[i],desc_querys[i],key_querys[n],desc_querys[n])
            goodmatch.append(onegoodmatch)
            match.append(onematch)
            n=n+1
        goodmatches.append(goodmatch)
        matches.append(match)


    matchidxs = []#特征匹配idx数组，N*N个格子，每个格子里面存着所有特征点对所对应的图片特征点idx
    list_kp_1s=[]#特征匹配数组中第一张图片的特征点像素坐标，N*N个格子（N为图片数量），每个格子里面存着许多对特征点;
    list_kp_2s=[]#特征匹配数组中第二张图片的特征点像素坐标
    # start = time.time()
    # for i in range(imgnum):
    #     n = 0
    #     matchidx = []
    #     list_kp_1 = []
    #     list_kp_2 = []
    #     while n < imgnum:
    #         one_list_kp_1, one_list_kp_2, one_matchidx = get_coordinates_from_matches(goodmatches[i][n], key_querys[i], key_querys[n])
    #         matchidx.append(one_matchidx)
    #         list_kp_1.append(one_list_kp_1)
    #         list_kp_2.append(one_list_kp_2)
    #         n=n+1
    #     matchidxs.append(matchidx)
    #     list_kp_1s.append(list_kp_1)
    #     list_kp_2s.append(list_kp_2)
    # end= time.time()
    # print(f"第一种：{end - start}")

    start = time.time()
    for i in range(imgnum):
        n = 0
        matchidx = []
        list_kp_1 = []
        list_kp_2 = []
        while n < imgnum:
            if n <= i:
                matchidx.append(0)
                list_kp_1.append(0)
                list_kp_2.append(0)
            else:
                one_list_kp_1, one_list_kp_2, one_matchidx = get_coordinates_from_matches(goodmatches[i][n], key_querys[i], key_querys[n])
                matchidx.append(one_matchidx)
                list_kp_1.append(one_list_kp_1)
                list_kp_2.append(one_list_kp_2)
            n=n+1
        matchidxs.append(matchidx)
        list_kp_1s.append(list_kp_1)
        list_kp_2s.append(list_kp_2)
    end = time.time()
    print(f"第二种：{end - start}")


    # imgidx1 = 0#特征匹配可视化，需要手动更改序号
    # imgidx2 = 1
    # knn_image = cv2.drawMatchesKnn(imgs[imgidx1], key_querys[imgidx1], imgs[imgidx2], key_querys[imgidx2], goodmatches[imgidx1][imgidx2], None)#展示对应点
    # plt.imshow(knn_image)
    # plt.show()



    return list_kp_1s, list_kp_2s,imgcolors,matchidxs
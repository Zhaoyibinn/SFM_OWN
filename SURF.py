import cv2
import numpy as np
import matplotlib.pyplot as plt



def get_coordinates_from_matches(matches: list, kp1: list, kp2: list):#返回坐标,匹配的idx

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
    surf = cv2.xfeatures2d_SURF.create(2000)
    # 找到关键点和描述符
    key_query, desc_query = surf.detectAndCompute(img, None)
    return key_query, desc_query

def surf_match(key_query1, desc_query1,key_query2, desc_query2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_query1, desc_query2, k=2)

    goodmatches=[]
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            goodmatches.append([m])

    # goodmatches = [[m] for m, n in matches if m.distance < 0.3 * n.distance]#次优比最优差0.7倍以上才可以
    return goodmatches,matches

def surf(imgpaths,imgnum):
    imgcolors = []
    for i in range(imgnum):
        imgcolors.append(cv2.imread(imgpaths[i]))

    imgs = []
    for i in range(imgnum):
        imgs.append(cv2.imread(imgpaths[i],0))
    height, width = imgs[0].shape

    k=1 #缩放系数
    for i in range(imgnum):
        imgs[i]= cv2.resize(imgs[i], (int(width/k), int(height/k)), interpolation=cv2.INTER_LINEAR)

    key_querys = []
    desc_querys = []
    for i in range(imgnum):
        key_query, desc_query = surf_detect(imgs[i])
        key_querys.append(key_query)
        desc_querys.append(desc_query)



    # detectimg = cv2.drawKeypoints(imgs[2], key_querys[2], imgs[2])#捕捉到的点画在图上
    # cv2.imshow('sp1',detectimg)
    # cv2.waitKey(0)
    # detectimg = cv2.drawKeypoints(imgs[3], key_querys[3], imgs[3])#捕捉到的点画在图上
    # cv2.imshow('sp2',detectimg)
    # cv2.waitKey(0)

    goodmatches = []
    matches = []

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

        # testonegoodmatch, testonematch = surf_match(key_querys[0], desc_querys[0], key_querys[1], desc_querys[1])


    matchidxs = []
    list_kp_1s=[]
    list_kp_2s=[]
    for i in range(imgnum):
        n = 0
        matchidx = []
        list_kp_1 = []
        list_kp_2 = []
        while n < imgnum:
            one_list_kp_1, one_list_kp_2, one_matchidx = get_coordinates_from_matches(goodmatches[i][n], key_querys[i], key_querys[n])
            matchidx.append(one_matchidx)
            list_kp_1.append(one_list_kp_1)
            list_kp_2.append(one_list_kp_2)
            n=n+1
        matchidxs.append(matchidx)
        list_kp_1s.append(list_kp_1)
        list_kp_2s.append(list_kp_2)



    # imgidx1 = 0
    # imgidx2 = 1
    # knn_image = cv2.drawMatchesKnn(imgs[imgidx1], key_querys[imgidx1], imgs[imgidx2], key_querys[imgidx2], goodmatches[imgidx1][imgidx2], None)#展示对应点
    # plt.imshow(knn_image)
    # plt.show()

    # for point in list_kp1:#用于测试坐标点
    #     cv2.circle(img1color,point, 2, (0, 0, 255),10)
    # cv2.imshow('sp1',img1color)
    # cv2.waitKey(0)
    # for point in list_kp2:
    #     cv2.circle(img2color,point, 2, (0, 0, 255),10)
    # cv2.imshow('sp1', img2color)
    # cv2.waitKey(0)


    # cv2.circle(img1color,list_kp1[50], 2, (0, 0, 255),10)
    # cv2.imshow('sp1',img1color)
    # cv2.waitKey(0)
    # cv2.circle(img2color,list_kp2[50], 2, (0, 0, 255),10)
    # cv2.imshow('sp1', img2color)
    # cv2.waitKey(0)


    # src_pts = np.float32([key_query1[m.queryIdx].pt for m in goodmatches])
    # dst_pts = np.float32([key_query2[m.trainIdx].pt for m in goodmatches])


    return list_kp_1s, list_kp_2s,imgcolors,matchidxs#直接返回坐标,返回的格式为：list_kp_1s[0][1]为第0张和第1张图片进行surf匹配之后，第一张图片的特征点坐标位置，为N*2的数组，list_kp_2s[0][1]则为第二张同理
                                                                        #matchidxs[0][1]则为第0张图和第1张图匹配时候的描述子序列，也为N*2
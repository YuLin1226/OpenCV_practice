# -*- coding:utf-8 -*-
import cv2 
import numpy as np 
from matplotlib import pyplot as plt
import math
import copy

def _transform_state(input_map):
    """
    Probabilities in OGMs will be transformed into 3 values: 0, 128, 255. \n
    Occupied Space:   0\n
    Unknown  Space: 128\n
    Free     Space: 255
    """

    r, c = np.shape(input_map)
    for i in range(r):
        for j in range(c):

            if input_map[i, j]  > 230:  # White >> Free Space
                input_map[i, j] = 255
            
            elif input_map[i, j] < 100: # Black >> Occupied Space
                input_map[i, j] = 0
            
            else:                       # Gray  >> Unknown Space
                input_map[i, j] = 255

    return input_map

def feature_detect_match(img1, img2, method="sift", delete_pairs=False):
    
    if method == "sift":
        siftDetector = cv2.xfeatures2d.SIFT_create()
        key_points_1, descriptor_1 = siftDetector.detectAndCompute(img1, None)
        key_points_2, descriptor_2 = siftDetector.detectAndCompute(img2, None)
    
    elif method == "surf":
        surf = cv2.xfeatures2d.SURF_create(400)
        key_points_1, descriptor_1 = surf.detectAndCompute(img1, None)
        key_points_2, descriptor_2 = surf.detectAndCompute(img2, None)
    
    elif method == "orb":        
        orb = cv2.ORB_create()
        key_points_1, descriptor_1 = orb.detectAndCompute(img1, None)
        key_points_2, descriptor_2 = orb.detectAndCompute(img2, None)
    
    elif method == "fast":        
        fast = cv2.FastFeatureDetector_create()
        key_points_1 = fast.detect(img1, None)
        key_points_2 = fast.detect(img2, None)
        
        br = cv2.BRISK_create()
        key_points_1, descriptor_1 = br.compute(img1,  key_points_1)
        key_points_2, descriptor_2 = br.compute(img2,  key_points_2)

    else:
        print("Method selection error, please input 'sift', 'surf', or 'orb'.")
        return

    bf = cv2.BFMatcher() # 創建暴力匹配對象，cv2.BFMatcher()；
    matches = bf.knnMatch(descriptor_1, descriptor_2, k=2) # 使用Matcher.knnMatch()獲得兩幅圖像的K個最佳匹配；
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance: #獲得的K個最佳匹配中取出來第一個和第二個，進行比值，比值小於0.75，則爲好的匹配點
            good.append([m])
            
    # roughly choose pairs by the slopes
    # pre_count = 0
    # for i in good:
    #     query_idx = i[0].queryIdx
    #     train_idx = i[0].trainIdx
    #     dy = key_points_2[train_idx].pt[1] - key_points_1[query_idx].pt[1]
    #     dx = key_points_2[train_idx].pt[0] - key_points_1[query_idx].pt[0]
    #     theta = math.atan2(dy, dx)
    #     good_pair = []    
    #     count = 0
    #     for j in good:
    #         query_idx_j = j[0].queryIdx
    #         train_idx_j = j[0].trainIdx
    #         dy_j = key_points_2[train_idx_j].pt[1] - key_points_1[query_idx_j].pt[1]
    #         dx_j = key_points_2[train_idx_j].pt[0] - key_points_1[query_idx_j].pt[0]
    #         theta_j = math.atan2(dy_j, dx_j)

    #         if abs(theta - theta_j) <= math.pi*5/180:
    #             good_pair.append(j)
    #             count += 1
                
    #     if count > pre_count:
    #         pre_count = count
    #         best_pair = copy.copy(good_pair)



    if delete_pairs:
        good_new = []
        train_list = []
        for i in range(len(good)):
            if good[i][0].trainIdx not in train_list:
                good_new.append(good[i])
                train_list.append(good[i][0].trainIdx)
        
        img3 = np.empty((600,600))
        img3 = cv2.drawMatchesKnn(img1, key_points_1, img2, key_points_2, good_new, img3, flags=2) #採用cv2.drawMatchesKnn()函數，在最佳匹配的點之間繪製直線

    else:
        img3 = np.empty((600,600))
        img3 = cv2.drawMatchesKnn(img1, key_points_1, img2, key_points_2, good, img3, flags=2) #採用cv2.drawMatchesKnn()函數，在最佳匹配的點之間繪製直線

    return img3


def dilate(img, L=1):

    img_r, img_c = np.shape(img)
    new_img = np.ones((img_r, img_c))*255
    black_pts = np.where(img==0)
    _, length = np.shape(black_pts)
    for i in range(length):
        x = black_pts[0][i]
        y = black_pts[1][i]
        new_img[x-L:x+L+1, y-L:y+L+1] = 0

    return new_img.astype(np.uint8)

def erode(img, L=1):

    img_r, img_c = np.shape(img)
    new_img = np.ones((img_r, img_c))*255
    black_pts = np.where(img==0)
    _, length = np.shape(black_pts)
    for i in range(length):
        x = black_pts[0][i]
        y = black_pts[1][i]
        if np.amax(img[x-L:x+L+1, y-L:y+L+1]) == 0:
            new_img[x,y] = 0

    return new_img.astype(np.uint8)

def find_contour(img1, img2):
    
    r,c = np.shape(img1)
    new_img = np.ones((r,c))*255
    for i in range(r):
        for j in range(c):
            if img1[i,j]!=img2[i,j]:
                new_img[i,j]=0
    return new_img.astype(np.uint8)


if __name__ == '__main__':
    img_1 = cv2.imread('Gazebo_test_map/tesst1.png')
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY)
    img_1 = _transform_state(input_map=img_1)
    img_1_ = dilate(img_1, L=2)
    img_1_de = erode(img_1_, L=1)
    img_1_ct = find_contour(img1=img_1_, img2=dilate(img=img_1, L=1))

    img_2 = cv2.imread('Gazebo_test_map/test0223_2.png')
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_RGB2GRAY) 
    img_2 = _transform_state(input_map=img_2)
    img_2_ = dilate(img_2, L=2)
    img_2_de = erode(img_2_, L=1)
    img_2_ct = find_contour(img1=img_2_, img2=dilate(img=img_2, L=1))

    img_sift = feature_detect_match(img1=img_1_ct, img2=img_2_ct, method="sift")
    # img_dilate_sift = feature_detect_match(img1=img_1_, img2=img_2_, method="sift")
    # img_dilate_erode_sift = feature_detect_match(img1=img_1_de, img2=img_2_de, method="sift")
    # img_surf = feature_detect_match(img1=img_1, img2=img_2, method="fast")
    # img_orb = feature_detect_match(img1=img_1, img2=img_2, method="orb")
    
    # fig, ax = plt.subplots(3, 1, figsize=(10, 8))

    # ax[0].imshow(img_sift, cmap='gray'), ax[0].set_title("sift")
    # ax[1].imshow(img_surf, cmap='gray'), ax[1].set_title("fast")
    # ax[2].imshow(img_orb, cmap='gray'), ax[2].set_title("orb")

    plt.figure()    
    # p1 = plt.subplot(311)
    plt.imshow(img_sift, cmap='gray')
    # p1.set_title('sift')

    # p2 = plt.subplot(312)
    # plt.imshow(img_dilate_sift, cmap='gray')
    # p2.set_title('dilate + sift')

    # p3 = plt.subplot(313)
    # plt.imshow(img_dilate_erode_sift, cmap='gray')
    # p3.set_title('dilate + erode + sift')
    # plt.tight_layout()

    # plt.figure()
    # pp1 = plt.subplot(221)
    # plt.imshow(img_1_, cmap='gray')
    # pp1.set_title('dilate')

    # pp2 = plt.subplot(222)
    # plt.imshow(img_1_de, cmap='gray')
    # pp2.set_title('dilate + erode')
    
    # pp3 = plt.subplot(223)
    # plt.imshow(img_2_, cmap='gray')
    # pp3.set_title('dilate')

    # pp4 = plt.subplot(224)
    # plt.imshow(img_2_de, cmap='gray')
    # pp4.set_title('dilate + erode')

    # plt.tight_layout()
    plt.show()
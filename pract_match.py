# -*- coding:utf-8 -*-
# URL: https://www.itread01.com/content/1547651741.html
# URL: https://www.twblogs.net/a/5c15f1e7bd9eee5e418425b7



import cv2
import numpy as np
from matplotlib import pyplot as plt

# img_1 = cv2.imread('occ.png')
# img_2 = cv2.imread('occ1.png')
img_1 = cv2.imread('map2.1.png')
img_2 = cv2.imread('map2.2.png')

# gray_1 = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY) # 灰階圖
# gray_2 = cv2.cvtColor(img_2, cv2.COLOR_RGB2GRAY) # 灰階圖

siftDetector = cv2.xfeatures2d.SIFT_create()
key_points_1, descriptor_1 = siftDetector.detectAndCompute(img_1, None)
key_points_2, descriptor_2 = siftDetector.detectAndCompute(img_2, None)

bf = cv2.BFMatcher() # 創建暴力匹配對象，cv2.BFMatcher()；
matches = bf.knnMatch(descriptor_1, descriptor_2, k=2) # 使用Matcher.knnMatch()獲得兩幅圖像的K個最佳匹配；


# matches class -> DMatch
# DMatch Attribute:
# 1. distance
# 2. imgIdx : always constant 0, useful when matching with multiple images.
# 3. queryIdx : train image index (first image)
# 4. trainIdx : query descriptor index (second image)


good = []
for m, n in matches:
    if m.distance < 0.65*n.distance: #獲得的K個最佳匹配中取出來第一個和第二個，進行比值，比值小於0.75，則爲好的匹配點
        good.append([m])
        # print(m.queryIdx, m.trainIdx, m.imgIdx)
        # print(n.queryIdx, n.trainIdx, n.imgIdx)

img_3 = np.empty((600,600))
img_3 = cv2.drawMatchesKnn(img_1, key_points_1, img_2, key_points_2, good[:20], img_3, flags=2) #採用cv2.drawMatchesKnn()函數，在最佳匹配的點之間繪製直線

xM_1 = key_points_1[good[0][0].queryIdx].pt[0]
xm_1 = key_points_1[good[0][0].queryIdx].pt[0]
yM_1 = key_points_1[good[0][0].queryIdx].pt[1] 
ym_1 = key_points_1[good[0][0].queryIdx].pt[1]

xM_2 = key_points_2[good[0][0].trainIdx].pt[0]
xm_2 = key_points_2[good[0][0].trainIdx].pt[0] 
yM_2 = key_points_2[good[0][0].trainIdx].pt[1] 
ym_2 = key_points_2[good[0][0].trainIdx].pt[1]
                            
for i in good[:20]:
    query_idx = i[0].queryIdx
    train_idx = i[0].trainIdx

    if key_points_1[query_idx].pt[0] > xM_1:
        xM_1 = key_points_1[query_idx].pt[0]
    elif key_points_1[query_idx].pt[0] < xm_1:
        xm_1 = key_points_1[query_idx].pt[0]

    if key_points_1[query_idx].pt[1] > yM_1:
        yM_1 = key_points_1[query_idx].pt[1]
    elif key_points_1[query_idx].pt[1] < ym_1:
        ym_1 = key_points_1[query_idx].pt[1]

    if key_points_2[train_idx].pt[0] > xM_2:
        xM_2 = key_points_2[train_idx].pt[0]
    elif key_points_2[train_idx].pt[0] < xm_2:
        xm_2 = key_points_2[train_idx].pt[0]

    if key_points_2[train_idx].pt[1] > yM_2:
        yM_2 = key_points_2[train_idx].pt[1]
    elif key_points_2[train_idx].pt[1] < ym_2:
        ym_2 = key_points_2[train_idx].pt[1]

dx, dy, _ = np.shape(img_1)
xM_2 += dy
xm_2 += dy

plt.imshow(img_3)
plt.plot([xM_1, xM_1, xm_1, xm_1, xM_1], [yM_1, ym_1, ym_1, yM_1, yM_1], color='red')
plt.plot([xM_2, xM_2, xm_2, xm_2, xM_2], [yM_2, ym_2, ym_2, yM_2, yM_2], color='red')
plt.show()







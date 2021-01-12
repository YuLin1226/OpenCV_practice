# -*- coding:utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import random


class Scale_Registration():
    def __init__(self):
        pass
    
    def _cal_distance(self, x1, y1, x2, y2):
        return ((x1-x2)**2 + (y1-y2)**2)**0.5


    def _ransac_find_scale(self, pts_set_1, pts_set_2, sigma, max_iter=1000):

        length, _ = np.shape(pts_set_1)
        best_ratio = 0
        total_inlier, pre_total_inlier = 0, 0

        for i in range(max_iter):
            
            index = random.sample(range(length), 2)
            x11 = pts_set_1[index[0], 0]
            y11 = pts_set_1[index[0], 1]
            x12 = pts_set_1[index[1], 0]
            y12 = pts_set_1[index[1], 1]
            
            x21 = pts_set_2[index[0], 0]
            y21 = pts_set_2[index[0], 1]
            x22 = pts_set_2[index[1], 0]
            y22 = pts_set_2[index[1], 1]

            dist_1 = self._cal_distance(x11, y11, x12, y12)
            dist_2 = self._cal_distance(x21, y21, x22, y22)

            mean_1 = [(x11+x12)/2, (y11+y12)/2]
            mean_2 = [(x21+x22)/2, (y21+y22)/2]

            ratio = dist_1 / dist_2
            
            for j in range(length):
                x1j = pts_set_1[j,0]
                y1j = pts_set_1[j,1]
                x2j = pts_set_2[j,0]
                y2j = pts_set_2[j,1]

                dist_1_j = self._cal_distance(mean_1[0], mean_1[1], x1j, y1j)
                dist_2_j = self._cal_distance(mean_2[0], mean_2[1], x2j, y2j)

                ratio_j = dist_1_j / dist_2_j

                if abs(ratio - ratio_j) < sigma:

                    total_inlier += 1

            if total_inlier > pre_total_inlier:
                
                pre_total_inlier = total_inlier
                best_ratio = ratio
            
            total_inlier = 0
        return best_ratio

def _translate_image(image, x, y):

    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

def _rotate_image(image, angle, center=None, scale=1.0):

    (h, w) = image.shape[:2]
    # If the center of the rotation is not defined, use the center of the image.
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def _state_process(img):
    r, c = np.shape(img)
    for i in range(r):
        for j in range(c):
            if img[i, j]  >0:
                img[i, j] = 255
            else:
                img[i, j] = 0
    return img

if __name__ == "__main__":
    
    img_1 = cv2.imread('map4.1.png')
    img_2 = cv2.imread('map4.3.png')
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY) # 灰階圖
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_RGB2GRAY) # 灰階圖
    img_1 = _state_process(img_1)
    img_2 = _state_process(img_2)    

    ratio = 1.75
    if ratio > 1:
        r, c = np.shape(img_1)
        img_3 = np.ones((math.ceil(r*ratio), math.ceil(c*ratio)))*255
        img_3[0:r, 0:c] = img_1
        img_3 = _rotate_image(image=img_3, angle=0, center=(0,0), scale=ratio).astype(np.uint8)
    else:
        img_3 = _rotate_image(image=img_1, angle=0, center=(0,0), scale=ratio)
    
    # print(np.amax(img_3), np.amin(img_3))
    siftDetector = cv2.xfeatures2d.SIFT_create()
    key_points_1, descriptor_1 = siftDetector.detectAndCompute(img_1, None)
    key_points_3, descriptor_3 = siftDetector.detectAndCompute(img_3, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor_1, descriptor_3, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    
    pts_1, pts_3 = [], []
    for i in good:
        query_idx = i.queryIdx
        train_idx = i.trainIdx

        pts_1.append([
            key_points_1[query_idx].pt[0],
            key_points_1[query_idx].pt[1],
            ])
        pts_3.append([
            key_points_3[train_idx].pt[0],
            key_points_3[train_idx].pt[1],
            ])
    
    pts1 = np.array(pts_1)
    pts3 = np.array(pts_3)

    # print(np.shape(pts1))

    # img_4 = np.empty((600,600))
    # img_4 = cv2.drawMatchesKnn(img_1, key_points_1, img_3, key_points_3, good[:20], img_4, flags=2)
    # plt.imshow(img_4, cmap='gray')
    # plt.show()
    
    for i in range(10):
        ratio_ransac = Scale_Registration()._ransac_find_scale(pts_set_1=pts1, pts_set_2=pts3, sigma=0.05, max_iter=100)
        print("ratio: %f"%ratio_ransac)
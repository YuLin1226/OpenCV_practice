# -*- coding:utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math




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

    ratio = 2
    if ratio > 1:
        r, c = np.shape(img_1)
        img_3 = np.ones((math.ceil(r*ratio), math.ceil(c*ratio)))*255
        img_3[0:r, 0:c] = img_1
        img_3 = _rotate_image(image=img_3, angle=0, center=(0,0), scale=ratio).astype(int)
    else:
        img_3 = _rotate_image(image=img_1, angle=0, center=(0,0), scale=ratio)
    
    # print(np.amax(img_3), np.amin(img_3))
    siftDetector = cv2.xfeatures2d.SIFT_create()
    # key_points_1, descriptor_1 = siftDetector.detectAndCompute(img_1, None)
    key_points_3, descriptor_3 = siftDetector.detectAndCompute(img_3, None)

    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(descriptor_1, descriptor_3, k=2)

    # good = []
    # for m, n in matches:
    #     if m.distance < 0.75*n.distance:
    #         good.append([m])

    # img_4 = np.empty((600,600))
    # img_4 = cv2.drawMatchesKnn(img_1, key_points_1, img_3, key_points_3, good[:20], img_4, flags=2)
    # plt.imshow(img_4, cmap='gray')
    # plt.show()

    
    # img_3 = cv2.drawKeypoints(img_3,
    #                         outImage=img_3,
    #                         keypoints=key_points_3, 
    #                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    #                         color= (51, 163, 236))
    # plt.imshow(img_3, cmap='gray')
    # plt.show()
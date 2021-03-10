# -*- coding:utf-8 -*-
# URL: https://www.itread01.com/content/1547651741.html



import cv2
import numpy as np
from matplotlib import pyplot as plt
from bresenham import bresenham
import copy

def _rotate_image(image, theta, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # If the center of the rotation is not defined, use the center of the image.
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, theta, scale)
    abs_cos = abs(M[0,0]) 
    abs_sin = abs(M[0,1])
    # find the new width and height bounds
    W = int(h * abs_sin + w * abs_cos)
    H = int(h * abs_cos + w * abs_sin)
    M[0, 2] += (W-w)/2
    M[1, 2] += (H-h)/2
    # theta = angle/math.pi*180
    rotated = cv2.warpAffine(image, M, (W, H))
    return rotated


def dilate(img, L=1):
    img_r, img_c = np.shape(img)
    new_img = np.ones((img_r, img_c))*0
    black_pts = np.where(img==255)
    _, length = np.shape(black_pts)
    for i in range(length):
        x = black_pts[0][i]
        y = black_pts[1][i]
        new_img[x-L:x+L+1, y-L:y+L+1] = 255
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

def de(img, L=1):
    img = dilate(img=img, L=L)
    # img = erode(img=img, L=L)
    return img

def test(img1, img2):
    r,c = np.shape(img1)
    new_img = np.ones((r,c))*255
    for i in range(r):
        for j in range(c):
            if img1[i,j]!=img2[i,j]:
                new_img[i,j]=0
    return new_img
            
if __name__=="__main__":

    # img = cv2.imread('map4.1.png')
    # img = cv2.imread('lena.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # 灰階圖

    img_gray = np.ones((500,500))*255
    pts = list(bresenham(130,130,360,130)) + list(bresenham(360,130,100,345))
    for i in range(len(pts)):
        img_gray[pts[i][0], pts[i][1]] = 0


    # img_gray = test(img1=de(img=img_gray, L=1) ,img2=img_gray)#de(img=img_gray, L=10))
    # img_gray[200:350, 200:250] = 0

    img_gray = img_gray.astype(np.uint8)
    
    r, c = np.shape(img_gray)
    for i in range(r):
        for j in range(c):
            if img_gray[i, j]  < 255:
                 img_gray[i, j] = 255
            else:
                img_gray[i, j] = 0

    img_gray = dilate(img_gray)
    img_rot = _rotate_image(img_gray, 30)
    img_rot1 = copy.copy(img_rot)
    # sift
    siftDetector = cv2.xfeatures2d.SIFT_create()
    key_points, descriptor = siftDetector.detectAndCompute(img_rot, None)
    img_rot1 = cv2.drawKeypoints(img_rot1,
                            outImage=img_rot1,
                            keypoints=key_points, 
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                            color= (255, 0, 0))
    plt.figure()
    plt.title('pixel calculated by interpolation.')
    plt.imshow(img_rot1, cmap='gray')

    # r, c = np.shape(img_rot)
    # for i in range(r):
    #     for j in range(c):
    #         if img_rot[i, j]  > 0:
    #              img_rot[i, j] = 255
    #         else:
    #             img_rot[i, j] = 0

    # # sift
    # siftDetector = cv2.xfeatures2d.SIFT_create()
    # key_points, descriptor = siftDetector.detectAndCompute(img_rot, None)
    # img_rot = cv2.drawKeypoints(img_rot,
    #                         outImage=img_rot,
    #                         keypoints=key_points, 
    #                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    #                         color= (255, 0, 0))
    # plt.figure()
    # plt.imshow(img_rot, cmap='gray')
    # plt.title('pixel changed to binary value.')
    plt.show()

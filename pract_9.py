# -*- coding:utf-8 -*-
# URL: https://www.itread01.com/content/1547651741.html



import cv2
import numpy as np
from matplotlib import pyplot as plt
from bresenham import bresenham



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
            

img = cv2.imread('map4.1.png')
# img = cv2.imread('lena.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # 灰階圖


img_gray = np.ones((500,500))*255
pts = list(bresenham(250,200,300,416)) + list(bresenham(250,200,50,146))

L = 10

for i in range(len(pts)):
    img_gray[pts[i][0], pts[i][1]] = 0
img_gray = test(img1=de(img=img_gray, L=1) ,img2=img_gray)#de(img=img_gray, L=10))
# img_gray[200:350, 200:250] = 0

img_gray = img_gray.astype(np.uint8)


r, c = np.shape(img_gray)
for i in range(r):
    for j in range(c):
        if img_gray[i, j]  < 254:
             img_gray[i, j] = 0
        else:
            img_gray[i, j] = 255


siftDetector = cv2.xfeatures2d.SIFT_create()
key_points, descriptor = siftDetector.detectAndCompute(img_gray, None)


img_gray = cv2.drawKeypoints(img_gray,
                        outImage=img_gray,
                        keypoints=key_points, 
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color= (51, 163, 236))


# plt.imshow(img, cmap='gray')
# plt.figure()
plt.imshow(img_gray, cmap='gray')
plt.show()

# cv2.imshow("sift",img)

# # Press any button to stop the running code.
# cv2.waitKey(0) 
# cv2.destroyAllWindows()
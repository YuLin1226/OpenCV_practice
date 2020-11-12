# coding:utf-8
__author__ = 'Microcosm'
# URL: https://www.itread01.com/content/1550009363.html


import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("lena.jpg",1)

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# sobel 運算元
# cv2.Sobel(src,ddepth,dx,dy,dst=None,ksize,scale,delta)
# dx = 1  對x方向求梯度
# dy = 1  對y方向求梯度
img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
img_sobel = cv2.addWeighted(img_sobel_x,0.5,img_sobel_y,0.5,0)


# Laplace 運算元
img_laplace = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)

# Canny 運算元
img_canny = cv2.Canny(img_gray, 100 , 150)

plt.subplot(231), plt.imshow(img_gray, "gray"), plt.title("Original")
plt.subplot(234), plt.imshow(img_sobel_x, "gray"), plt.title("Sobel_x")
plt.subplot(235), plt.imshow(img_sobel_y, "gray"), plt.title("Sobel_y")
plt.subplot(232), plt.imshow(img_laplace,  "gray"), plt.title("Laplace")
plt.subplot(233), plt.imshow(img_canny, "gray"), plt.title("Canny")
plt.subplot(236), plt.imshow(img_sobel, "gray"), plt.title("Sobel")
plt.show()

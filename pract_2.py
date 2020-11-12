# coding=utf-8
__author__ = 'Microcosm'
# URL: https://www.itread01.com/content/1546866750.html
print("影象通道拆分")

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("lena.jpg")

# 獲取影象的尺寸及通道數
width, height, ch = img.shape
print("Width: %i , Height: %i , Channel: %i" %(width, height, ch))
print("Image Size Number = Width x Height x Channel = %i" %img.size)  # 列印畫素數目

# 彩色通道的拆分與合併，速度較慢
b, g, r = cv2.split(img)
img = cv2.merge([b, g, r])

# 採用類似 matlab 中的下標索引方式，速度比較快
# 分離彩色通道，順序是BGR,與 matlab 不同
B = img[:, :, 0]
G = img[:, :, 1]
R = img[:, :, 2]

# 調整RGB通道的順序
img_RGB = img.copy()
img_RGB[:, :, 0] = R
img_RGB[:, :, 1] = G
img_RGB[:, :, 2] = B

# 採用類似 matlab 顯示多個座標軸的形式並行顯示，比較方便
# 由於 matlab 是按照RGB順序顯示的，所以需要將原影象的順序重調一下
plt.subplot(231), plt.imshow(img,'gray'), plt.title("img_BGR")
plt.subplot(232), plt.imshow(img_RGB, "gray"), plt.title("Img_RGB")
plt.subplot(233), plt.imshow(B, "gray"), plt.title("B")
plt.subplot(234), plt.imshow(G, "gray"), plt.title("G")
plt.subplot(235), plt.imshow(R, "gray"), plt.title("R")
plt.show()
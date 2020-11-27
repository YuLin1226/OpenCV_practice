
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("map4.1.png")
# img_gray = np.array([
#     [-1,-1,1,-1,-1],
#     [-1,-1,-1,-1,-1],
#     [0,0,0,0,0],
#     [0,0,0,0,0],
#     [1,1,1,1,1],
#     [1,1,1,1,1],
# ])

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# print(np.amax(img_gray))

r, c = np.shape(img_gray)
for i in range(r):
    for j in range(c):
        if img_gray[i, j]  < 254:
             img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1

# sobel 運算元
# cv2.Sobel(src,ddepth,dx,dy,dst=None,ksize,scale,delta)
# dx = 1  對x方向求梯度
# dy = 1  對y方向求梯度
img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

mag, angle = cv2.cartToPolar(img_sobel_x, img_sobel_y, angleInDegrees=True)


plt.imshow(img_gray,cmap='gray')

r, c = np.shape(angle)
angle_1d = np.reshape(angle, (1, r*c))

angle_list = []
for i in range(r*c):
    k = int(angle_1d[0,i]/22.5)
    angle_list.append(k)

angle_list = np.array(angle_list)
plt.figure()
plt.hist(angle_list, bins='auto')
plt.show()

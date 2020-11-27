# -*- coding:utf-8 -*-
# URL: https://www.itread01.com/content/1547651741.html



import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('map4.1.png')
# img = cv2.imread('lena.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # 灰階圖

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
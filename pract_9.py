# -*- coding:utf-8 -*-
# URL: https://www.itread01.com/content/1547651741.html



import cv2
import numpy as np

img = cv2.imread('lena.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # 灰階圖

siftDetector = cv2.xfeatures2d.SIFT_create()
key_points, descriptor = siftDetector.detectAndCompute(gray, None)
print(len(key_points))
print(descriptor.shape) # 每個特徵點的128個特徵值

img = cv2.drawKeypoints(img,
                        outImage=img,
                        keypoints=key_points, 
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color= (51, 163, 236))



# cv2.imshow("sift",img)

# # Press any button to stop the running code.
# cv2.waitKey(0) 
# cv2.destroyAllWindows()
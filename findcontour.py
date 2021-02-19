# -*- coding: UTF-8 -*- 

import cv2  
import numpy as np
from matplotlib import pyplot as plt
import copy

def _transform_state(input_map, state_range=[230, 100]):
    """
    Probabilities in OGMs will be transformed into 3 values: 0, 128, 255. \n
    Occupied Space:   0\n
    Unknown  Space: 128\n
    Free     Space: 255
    """

    r, c = np.shape(input_map)
    for i in range(r):
        for j in range(c):

            if input_map[i, j]  > state_range[0]:  # White >> Free Space
                input_map[i, j] = 255
            
            elif input_map[i, j] < state_range[1]: # Black >> Occupied Space
                input_map[i, j] = 0
            
            else:                       # Gray  >> Unknown Space
                input_map[i, j] = 255

    return input_map

def _extend_black_pixel(input_map):

    black_pts = np.where(input_map==0)
    r, c = np.shape(black_pts)
    for i in range(c):
        x = black_pts[0][i]
        y = black_pts[1][i]
        input_map[x-5:x+6, y-5:y+6] = 0

    return input_map

img = cv2.imread("Gazebo_test_map//tesst.png")  
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
img_binary = _transform_state(input_map=img_gray)


    




ret, binary = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
_, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
# cv2.drawContours(img,contours,-1,(0,0,255),3)  

a = np.ones(np.shape(img_gray))*255

for i in range(len(contours)-1):
# for i in range(1):

    x = contours[i+1][:,0,0]
    y = contours[i+1][:,0,1]
    
    vertex = np.vstack((x,y)).T
    # print(np.shape(vertex))



    # triangle = np.array([ [10,30], [40,80], [10,90] ], np.int32) #注意xy座標
    cv2.fillConvexPoly(a, vertex, 0)




plt.imshow(a, cmap='gray')
# plt.subplot(211)
# plt.imshow(gray, cmap='gray')
# plt.subplot(212)
# plt.imshow(gray_1, cmap='gray')
plt.show()
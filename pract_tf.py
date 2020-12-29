import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt

def translate(image, x, y):

    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return shifted

def rotate(image, angle, center=None, scale=1.0):

    (h, w) = image.shape[:2]
 
    # If the center of the rotation is not defined, use the center of the image.
    if center is None:
        center = (w / 2, h / 2)
 
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
 
    return rotated

if __name__ == "__main__":
    
    img = cv2.imread('map4.1.png')
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # 灰階圖

    r,c = np.shape(img_gray)
    
    img_empty = np.zeros((3*r, 3*c))
    img_empty[r:2*r, c:c*2]=img_gray
    print(r,c)
    img_rot = rotate(image=img_empty, angle=0, center=(c,r), scale=1)
    img_rot1 = rotate(image=img_empty, angle=45, center=(c,r), scale=1)

    # img_rot = rotate(image=img_gray, angle=45, center=None, scale=0.5)
    # img_rot = rotate(image=img_rot, angle=0, center=None, scale=2)
    
    plt.subplot(1,2,1), plt.imshow(img_rot, cmap='gray'), plt.scatter([c],[r], s=10, c='red')
    plt.subplot(1,2,2), plt.imshow(img_rot1, cmap='gray'), plt.scatter([c],[r], s=10, c='red')
    plt.show()

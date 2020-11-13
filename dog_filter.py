import cv2
import numpy as np
from matplotlib import pyplot as plt


def DoG(img, kernel_size_1 = 77, kernel_size_2 = 13):
    
    #run a 5x5 gaussian blur then a 3x3 gaussian blr
    blur_1 = cv2.GaussianBlur(img,(kernel_size_1, kernel_size_1),0)
    blur_2 = cv2.GaussianBlur(img,(kernel_size_2, kernel_size_2),0)
    DoGim = blur_1 - blur_2
    
    return DoGim, blur_1, blur_2



img = cv2.imread("lena.jpg",1)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_dog, blur_1, blur_2 = DoG(img_gray)

plt.subplot(131), plt.imshow(img_dog, "gray")
plt.subplot(132), plt.imshow(blur_1, "gray")
plt.subplot(133), plt.imshow(blur_2, "gray")
plt.show()

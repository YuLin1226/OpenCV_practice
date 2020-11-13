import cv2
import numpy as np
from matplotlib import pyplot as plt


def DoG(img):
    
    #run a 5x5 gaussian blur then a 3x3 gaussian blr
    blur_1 = cv2.GaussianBlur(img,(15,15),0)
    blur_2 = cv2.GaussianBlur(img,(13,13),0)
    DoGim = blur_1 - blur_2
    
    return DoGim



img = cv2.imread("lena.jpg",1)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_dog = DoG(img_gray)

plt.imshow(img_dog, "gray")
plt.show()

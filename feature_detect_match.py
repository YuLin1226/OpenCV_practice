# -*- coding:utf-8 -*-
import cv2 
import numpy as np 
from matplotlib import pyplot as plt


def feature_detect_match(img1, img2, method="sift"):
    
    if method == "sift":
        siftDetector = cv2.xfeatures2d.SIFT_create()
        key_points_1, descriptor_1 = siftDetector.detectAndCompute(img1, None)
        key_points_2, descriptor_2 = siftDetector.detectAndCompute(img2, None)
    
    elif method == "surf":
        surf = cv2.xfeatures2d.SURF_create(400)
        key_points_1, descriptor_1 = surf.detectAndCompute(img1, None)
        key_points_2, descriptor_2 = surf.detectAndCompute(img2, None)
    
    elif method == "orb":        
        orb = cv2.ORB_create()
        key_points_1, descriptor_1 = orb.detectAndCompute(img1, None)
        key_points_2, descriptor_2 = orb.detectAndCompute(img2, None)
    
    elif method == "fast":        
        fast = cv2.FastFeatureDetector_create()
        key_points_1 = fast.detect(img1, None)
        key_points_2 = fast.detect(img2, None)
        
        br = cv2.BRISK_create()
        key_points_1, descriptor_1 = br.compute(img1,  key_points_1)
        key_points_2, descriptor_2 = br.compute(img2,  key_points_2)

    else:
        print("Method selection error, please input 'sift', 'surf', or 'orb'.")
        return

    bf = cv2.BFMatcher() # 創建暴力匹配對象，cv2.BFMatcher()；
    matches = bf.knnMatch(descriptor_1, descriptor_2, k=2) # 使用Matcher.knnMatch()獲得兩幅圖像的K個最佳匹配；
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance: #獲得的K個最佳匹配中取出來第一個和第二個，進行比值，比值小於0.75，則爲好的匹配點
            good.append([m])
    img3 = np.empty((600,600))
    img3 = cv2.drawMatchesKnn(img1, key_points_1, img2, key_points_2, good, img3, flags=2) #採用cv2.drawMatchesKnn()函數，在最佳匹配的點之間繪製直線

    return img3






if __name__ == '__main__':
    img_1 = cv2.imread('Gazebo_test_map/tesst2.png')
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY) 
    img_2 = cv2.imread('Gazebo_test_map/tesst1.png')
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_RGB2GRAY) 

    img_sift = feature_detect_match(img1=img_1, img2=img_2, method="sift")
    img_surf = feature_detect_match(img1=img_1, img2=img_2, method="fast")
    img_orb = feature_detect_match(img1=img_1, img2=img_2, method="orb")
    
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))

    ax[0].imshow(img_sift, cmap='gray'), ax[0].set_title("sift")
    ax[1].imshow(img_surf, cmap='gray'), ax[1].set_title("fast")
    ax[2].imshow(img_orb, cmap='gray'), ax[2].set_title("orb")
    
    plt.show()
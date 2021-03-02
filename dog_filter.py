# -*- coding:utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt


def _transform_state(input_map):
    """
    Probabilities in OGMs will be transformed into 3 values: 0, 128, 255. \n
    Occupied Space:   0\n
    Unknown  Space: 128\n
    Free     Space: 255
    """

    r, c = np.shape(input_map)
    for i in range(r):
        for j in range(c):

            if input_map[i, j]  > 230:  # White >> Free Space
                input_map[i, j] = 255
            
            elif input_map[i, j] < 100: # Black >> Occupied Space
                input_map[i, j] = 0
            
            else:                       # Gray  >> Unknown Space
                input_map[i, j] = 255

    return input_map

def DoG(img, kernel_size_1 = 77, kernel_size_2 = 13):
    
    #run a 5x5 gaussian blur then a 3x3 gaussian blr
    blur_1 = cv2.GaussianBlur(img,(kernel_size_1, kernel_size_1),0)
    blur_2 = cv2.GaussianBlur(img,(kernel_size_2, kernel_size_2),0)
    DoGim = blur_1 - blur_2
    
    return DoGim, blur_1, blur_2



def draw_sift(img):

    siftDetector = cv2.xfeatures2d.SIFT_create()
    key_points, descriptor = siftDetector.detectAndCompute(img, None)
    img = cv2.drawKeypoints(img,
                            outImage=img,
                            keypoints=key_points, 
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                            color= (51, 163, 236))

    return img

def feature_detect_match(img1, img2, method="sift", delete_pairs=False):
    
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
            
    # roughly choose pairs by the slopes
    # pre_count = 0
    # for i in good:
    #     query_idx = i[0].queryIdx
    #     train_idx = i[0].trainIdx
    #     dy = key_points_2[train_idx].pt[1] - key_points_1[query_idx].pt[1]
    #     dx = key_points_2[train_idx].pt[0] - key_points_1[query_idx].pt[0]
    #     theta = math.atan2(dy, dx)
    #     good_pair = []    
    #     count = 0
    #     for j in good:
    #         query_idx_j = j[0].queryIdx
    #         train_idx_j = j[0].trainIdx
    #         dy_j = key_points_2[train_idx_j].pt[1] - key_points_1[query_idx_j].pt[1]
    #         dx_j = key_points_2[train_idx_j].pt[0] - key_points_1[query_idx_j].pt[0]
    #         theta_j = math.atan2(dy_j, dx_j)

    #         if abs(theta - theta_j) <= math.pi*5/180:
    #             good_pair.append(j)
    #             count += 1
                
    #     if count > pre_count:
    #         pre_count = count
    #         best_pair = copy.copy(good_pair)



    if delete_pairs:
        good_new = []
        train_list = []
        for i in range(len(good)):
            if good[i][0].trainIdx not in train_list:
                good_new.append(good[i])
                train_list.append(good[i][0].trainIdx)
        
        img3 = np.empty((600,600))
        img3 = cv2.drawMatchesKnn(img1, key_points_1, img2, key_points_2, good_new, img3, flags=2) #採用cv2.drawMatchesKnn()函數，在最佳匹配的點之間繪製直線

    else:
        img3 = np.empty((600,600))
        img3 = cv2.drawMatchesKnn(img1, key_points_1, img2, key_points_2, good, img3, flags=2) #採用cv2.drawMatchesKnn()函數，在最佳匹配的點之間繪製直線

    return img3




img = cv2.imread("Gazebo_test_map/test0223_2.png",1)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_gray = _transform_state(img_gray)
img_dog, blur_1, blur_2 = DoG(img_gray)
img_sift = draw_sift(img_dog)

img1 = cv2.imread("Gazebo_test_map/tesst1.png",1)
img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
img1_gray = _transform_state(img1_gray)
img1_dog, blur_11, blur_12 = DoG(img1_gray)
img1_sift = draw_sift(img1_dog)

img3 = feature_detect_match(img1_gray, img_gray)
img4 = feature_detect_match(img_gray, img1_gray)

# plt.subplot(131), plt.imshow(img_sift, "gray")
# plt.subplot(132), plt.imshow(img1_sift, "gray")
plt.subplot(211), plt.imshow(img3, "gray")
plt.subplot(212), plt.imshow(img4, "gray")

# plt.subplot(132), plt.imshow(blur_1, "gray")
# plt.subplot(133), plt.imshow(blur_2, "gray")
plt.show()

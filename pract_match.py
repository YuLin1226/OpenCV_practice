# -*- coding:utf-8 -*-
# URL: https://www.itread01.com/content/1547651741.html
# URL: https://www.twblogs.net/a/5c15f1e7bd9eee5e418425b7



import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from sklearn.neighbors import NearestNeighbors # For ICP

# Class
class ScanMatching():

    def __init__(self):
        pass


    def euclidean_distance(self, point1, point2):
        """
        Euclidean distance between two points.
        :param point1: the first point as a tuple (a_1, a_2, ..., a_n)
        :param point2: the second point as a tuple (b_1, b_2, ..., b_n)
        :return: the Euclidean distance
        """
        a = np.array(point1)
        b = np.array(point2)

        return np.linalg.norm(a - b, ord=2)


    def point_based_matching(self, point_pairs):
        """
        This function is based on the paper "Robot Pose Estimation in Unknown Environments by Matching 2D Range Scans"
        by F. Lu and E. Milios.

        :param point_pairs: the matched point pairs [((x1, y1), (x1', y1')), ..., ((xi, yi), (xi', yi')), ...]
        :return: the rotation angle and the 2D translation (x, y) to be applied for matching the given pairs of points
        """

        x_mean = 0
        y_mean = 0
        xp_mean = 0
        yp_mean = 0
        n = len(point_pairs)

        if n == 0:
            return None, None, None

        for pair in point_pairs:

            (x, y), (xp, yp) = pair

            x_mean += x
            y_mean += y
            xp_mean += xp
            yp_mean += yp

        x_mean /= n
        y_mean /= n
        xp_mean /= n
        yp_mean /= n

        s_x_xp = 0
        s_y_yp = 0
        s_x_yp = 0
        s_y_xp = 0
        for pair in point_pairs:

            (x, y), (xp, yp) = pair

            s_x_xp += (x - x_mean)*(xp - xp_mean)
            s_y_yp += (y - y_mean)*(yp - yp_mean)
            s_x_yp += (x - x_mean)*(yp - yp_mean)
            s_y_xp += (y - y_mean)*(xp - xp_mean)

        rot_angle = math.atan2(s_x_yp - s_y_xp, s_x_xp + s_y_yp)
        translation_x = xp_mean - (x_mean*math.cos(rot_angle) - y_mean*math.sin(rot_angle))
        translation_y = yp_mean - (x_mean*math.sin(rot_angle) + y_mean*math.cos(rot_angle))

        return rot_angle, translation_x, translation_y


    def icp(self, reference_points, points, max_iterations=2000, distance_threshold=300, convergence_translation_threshold=1,
            convergence_rotation_threshold=1e-3, point_pairs_threshold=600, verbose=False):
        """
        An implementation of the Iterative Closest Point algorithm that matches a set of M 2D points to another set
        of N 2D (reference) points.

        :param reference_points: the reference point set as a numpy array (N x 2)
        :param points: the point that should be aligned to the reference_points set as a numpy array (M x 2)
        :param max_iterations: the maximum number of iteration to be executed
        :param distance_threshold: the distance threshold between two points in order to be considered as a pair
        :param convergence_translation_threshold: the threshold for the translation parameters (x and y) for the
                                                transformation to be considered converged
        :param convergence_rotation_threshold: the threshold for the rotation angle (in rad) for the transformation
                                                to be considered converged
        :param point_pairs_threshold: the minimum number of point pairs the should exist
        :param verbose: whether to print informative messages about the process (default: False)
        :return: the transformation history as a list of numpy arrays containing the rotation (R) and translation (T)
                transformation in each iteration in the format [R | T] and the aligned points as a numpy array M x 2
        """

        transformation_history = []

        x, y, yaw = 0, 0, 0

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(reference_points)
        
        for iter_num in range(max_iterations):
            if verbose:
                print('------ iteration', iter_num, '------')

            closest_point_pairs = []  # list of point correspondences for closest point rule

            distances, indices = nbrs.kneighbors(points)
            for nn_index in range(len(distances)):
                if distances[nn_index][0] < distance_threshold:
                    closest_point_pairs.append((points[nn_index], reference_points[indices[nn_index][0]]))

            # if only few point pairs, stop process
            if verbose:
                print('number of pairs found:', len(closest_point_pairs))
            if len(closest_point_pairs) < point_pairs_threshold:
                if verbose:
                    print('No better solution can be found (very few point pairs)!')
                break

            # compute translation and rotation using point correspondences
            closest_rot_angle, closest_translation_x, closest_translation_y = self.point_based_matching(closest_point_pairs)
            if closest_rot_angle is not None:
                if verbose:
                    print('Rotation:', math.degrees(closest_rot_angle), 'degrees')
                    print('Translation:', closest_translation_x, closest_translation_y)
            if closest_rot_angle is None or closest_translation_x is None or closest_translation_y is None:
                if verbose:
                    print('No better solution can be found!')
                break

            # transform 'points' (using the calculated rotation and translation)
            c, s = math.cos(closest_rot_angle), math.sin(closest_rot_angle)
            rot = np.array([[c, -s],
                            [s, c]])
            aligned_points = np.dot(points, rot.T)
            aligned_points[:, 0] += closest_translation_x
            aligned_points[:, 1] += closest_translation_y

            # update 'points' for the next iteration
            points = aligned_points

            # update transformation history
            transformation_history.append(np.vstack((np.hstack( (rot, np.array([[closest_translation_x], [closest_translation_y]]) )), np.array([0,0,1]))))

            yaw += closest_rot_angle
            x += closest_translation_x
            y += closest_translation_y

            # check convergence
            if (abs(closest_rot_angle) < convergence_rotation_threshold) \
                    and (abs(closest_translation_x) < convergence_translation_threshold) \
                    and (abs(closest_translation_y) < convergence_translation_threshold):
                if verbose:
                    print('Converged!')
                break

        return transformation_history, points, [x, y, yaw]


def _rot(theta, vector):
    
    '''
    theta (rad)
    vector (Nx2)
    '''

    R = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta) ]
    ])

    return R.dot(vector.T).T


def _enlarge_map(input_map):

    r, c = np.shape(input_map)
    map_ext = np.ones((3*r, 3*c))*128
    map_ext[r:r+r, c:c+c] = input_map
    map_ext = map_ext.astype(np.uint8)
    return map_ext

def _rotate_image(image, angle, center=None, scale=1.0):

    (h, w) = image.shape[:2]
    # If the center of the rotation is not defined, use the center of the image.
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


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
                input_map[i, j] = 128

    return input_map

# First set of the image.
# img_1 = cv2.imread('occ.png')
# img_2 = cv2.imread('occ1.png')

# Second set of the image.
# img_1 = cv2.imread('map2.1.png')
# img_2 = cv2.imread('map2.2.png')

# Third set of the image.
# img_1 = cv2.imread('map4.1.png')
# # img_2 = cv2.imread('map4.2.png')
# img_2 = cv2.imread('map4.3.png')

# Third set of the image.
img_1 = cv2.imread('Gazebo_test_map//tesst1.png')
img_2 = cv2.imread('Gazebo_test_map//test0223_2.png')



img_1 = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY) # 灰階圖
img_2 = cv2.cvtColor(img_2, cv2.COLOR_RGB2GRAY) # 灰階圖


img_1 = _transform_state(input_map=img_1)
img_2 = _transform_state(input_map=img_2)



# orb = cv2.ORB_create()
# key_points_1, descriptor_1 = orb.detectAndCompute(img_1, None)
# key_points_2, descriptor_2 = orb.detectAndCompute(img_2, None)


siftDetector = cv2.xfeatures2d.SIFT_create()
key_points_1, descriptor_1 = siftDetector.detectAndCompute(img_1, None)
key_points_2, descriptor_2 = siftDetector.detectAndCompute(img_2, None)
# descriptor 內含 n 個特徵 (列)，每個特徵含有 128 個bin value (行)。
# key_points class 可查閱 CV2.KeyPoint()
# 捕捉特徵座標使用 key_point[n].pt[0 or 1], 0代表x, 1代表y

img_1_sift = cv2.drawKeypoints(img_1,
                        outImage=img_1,
                        keypoints=key_points_1, 
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color= (51, 163, 236))

plt.figure()
ax = plt.gca()      
plt.margins(0, 0)        
ax.imshow(img_1_sift, cmap='gray')
ax.xaxis.set_ticks_position('top') 
# ax.set_xlim(725,1145)
# ax.set_ylim(715,1120)
# ax.invert_yaxis()
if 1:
    plt.savefig("4_SIFT_result_1.png", bbox_inches='tight')


img_2_sift = cv2.drawKeypoints(img_2,
                        outImage=img_2,
                        keypoints=key_points_2, 
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color= (51, 163, 236))

plt.figure()
ax = plt.gca()      
plt.margins(0, 0)        
ax.imshow(img_2_sift, cmap='gray')
ax.xaxis.set_ticks_position('top') 
# ax.set_xlim(725,1145)
# ax.set_ylim(715,1120)
# ax.invert_yaxis()
if 1:
    plt.savefig("4_SIFT_result_2.png", bbox_inches='tight')


bf = cv2.BFMatcher() # 創建暴力匹配對象，cv2.BFMatcher()；
matches = bf.knnMatch(descriptor_1, descriptor_2, k=2) # 使用Matcher.knnMatch()獲得兩幅圖像的K個最佳匹配；
# bf.knnMatch 會輸出 descriptor_2 中對應 descriptor_1 的最佳匹配。
# 也就是說，matches 長度會與 descriptor_1 相同，matches 每一列的元素中含有K個來自 descriptor_2 的最佳匹配 (照相似度排序)。
print(len(descriptor_1), len(descriptor_2), len(matches))
print(matches[1][0].distance, matches[1][0].imgIdx, matches[1][0].queryIdx, matches[1][0].trainIdx)
print(matches[1][1].distance, matches[1][1].imgIdx, matches[1][1].queryIdx, matches[1][1].trainIdx)

# matches class -> DMatch
# DMatch Attribute:
# 1. distance : similarity, the smaller the more similar.
# 2. imgIdx : always constant 0, useful when matching with multiple images.
# 3. queryIdx : train image index (first image)，descriptor_1中的特徵
# 4. trainIdx : query descriptor index (second image)，，descriptor_2中的特徵


good = []
for m, n in matches:
    if m.distance < 0.5*n.distance: #獲得的K個最佳匹配中取出來第一個和第二個，進行比值，比值小於0.75，則爲好的匹配點
        good.append([m])
        # print(m.queryIdx, m.trainIdx, m.imgIdx)
        # print(n.queryIdx, n.trainIdx, n.imgIdx)

img_3 = np.empty((600,600))
img_3 = cv2.drawMatchesKnn(img_1, key_points_1, img_2, key_points_2, good, img_3, flags=2) #採用cv2.drawMatchesKnn()函數，在最佳匹配的點之間繪製直線

plt.figure()
ax = plt.gca()      
plt.margins(0, 0)        
ax.imshow(img_3, cmap='gray')
ax.xaxis.set_ticks_position('top') 
# ax.set_xlim(725,1145)
# ax.set_ylim(715,1120)
# ax.invert_yaxis()
if 1:
    plt.savefig("4_SIFT_match_test_1.png", bbox_inches='tight')





# xM_1 = key_points_1[good[0][0].queryIdx].pt[0]
# xm_1 = key_points_1[good[0][0].queryIdx].pt[0]
# yM_1 = key_points_1[good[0][0].queryIdx].pt[1] 
# ym_1 = key_points_1[good[0][0].queryIdx].pt[1]

# xM_2 = key_points_2[good[0][0].trainIdx].pt[0]
# xm_2 = key_points_2[good[0][0].trainIdx].pt[0] 
# yM_2 = key_points_2[good[0][0].trainIdx].pt[1] 
# ym_2 = key_points_2[good[0][0].trainIdx].pt[1]
                            
# for i in good[:20]:
#     query_idx = i[0].queryIdx
#     train_idx = i[0].trainIdx

#     if key_points_1[query_idx].pt[0] > xM_1:
#         xM_1 = key_points_1[query_idx].pt[0]
#     elif key_points_1[query_idx].pt[0] < xm_1:
#         xm_1 = key_points_1[query_idx].pt[0]

#     if key_points_1[query_idx].pt[1] > yM_1:
#         yM_1 = key_points_1[query_idx].pt[1]
#     elif key_points_1[query_idx].pt[1] < ym_1:
#         ym_1 = key_points_1[query_idx].pt[1]

#     if key_points_2[train_idx].pt[0] > xM_2:
#         xM_2 = key_points_2[train_idx].pt[0]
#     elif key_points_2[train_idx].pt[0] < xm_2:
#         xm_2 = key_points_2[train_idx].pt[0]

#     if key_points_2[train_idx].pt[1] > yM_2:
#         yM_2 = key_points_2[train_idx].pt[1]
#     elif key_points_2[train_idx].pt[1] < ym_2:
#         ym_2 = key_points_2[train_idx].pt[1]

# dx, dy = np.shape(img_1)
# xM_2 += dy
# xm_2 += dy

# Draw match result and red rectangle.
# plt.figure()
# plt.imshow(img_3)
# plt.plot([xM_1, xM_1, xm_1, xm_1, xM_1], [yM_1, ym_1, ym_1, yM_1, yM_1], color='red')
# plt.plot([xM_2, xM_2, xm_2, xm_2, xM_2], [yM_2, ym_2, ym_2, yM_2, yM_2], color='red')

# overlap_in_img_1 = np.zeros((int(xM_1),int(yM_1)))
# overlap_in_img_2 = np.zeros((int(xM_2),int(yM_2)))
# overlap_in_img_1 = img_1[int(ym_1)-50 : int(yM_1)+50 , int(xm_1)-50 : int(xM_1)+50]
# overlap_in_img_2 = img_2[int(ym_2)-50 : int(yM_2)+50 , int(xm_2-dy)-50 : int(xM_2-dy)+50]

# # Extract the red rectangles.
# plt.figure()
# plt.subplot(211), plt.imshow(overlap_in_img_1 ,cmap='gray')
# plt.subplot(212), plt.imshow(overlap_in_img_2 ,cmap='gray')
# # plt.show()


# occupied_list_1, occupied_list_2 = [], []

# # collect occupied points.
# for i in range(np.shape(overlap_in_img_1)[0]):
#     for j in range(np.shape(overlap_in_img_1)[1]):
#         if overlap_in_img_1[i,j] == 0:
#             occupied_list_1.append([i,j])

# for i in range(np.shape(overlap_in_img_2)[0]):
#     for j in range(np.shape(overlap_in_img_2)[1]):
#         if overlap_in_img_2[i,j] == 0:
#             occupied_list_2.append([i,j])

# # pre-rotation : move the center
# occupied_ndarray_1 = np.array(occupied_list_1)
# # dx = ( max(occupied_ndarray_1[:,0]) + min(occupied_ndarray_1[:,0]) )/2
# # dy = ( max(occupied_ndarray_1[:,1]) + min(occupied_ndarray_1[:,1]) )/2
# dx = np.mean(occupied_ndarray_1[:,0])
# dy = np.mean(occupied_ndarray_1[:,1])
# occupied_ndarray_1[:,0] -= int(dx)
# occupied_ndarray_1[:,1] -= int(dy)

# occupied_ndarray_2 = np.array(occupied_list_2)
# # dx = ( max(occupied_ndarray_2[:,0]) + min(occupied_ndarray_2[:,0]) )/2
# # dy = ( max(occupied_ndarray_2[:,1]) + min(occupied_ndarray_2[:,1]) )/2
# dx = np.mean(occupied_ndarray_2[:,0])
# dy = np.mean(occupied_ndarray_2[:,1])
# occupied_ndarray_2[:,0] -= int(dx)
# occupied_ndarray_2[:,1] -= int(dy)


# occupied_ndarray_2_ = _rot(math.pi/180*60 , occupied_ndarray_2)

# SM = ScanMatching()
# _, pts, [x, y, yaw] = SM.icp(occupied_ndarray_1, occupied_ndarray_2_)
# print("translation_(x, y) : (%f, %f) "%(x,y))
# print("angle : %f degree"%(yaw/math.pi * 180))


# plt.figure()
# p1 = plt.scatter(occupied_ndarray_1[:,0], occupied_ndarray_1[:,1], c='black', s=10)
# p2 = plt.scatter(occupied_ndarray_2_[:,0], occupied_ndarray_2_[:,1], c='blue', s=5)
# p3 = plt.scatter(pts[:,0], pts[:,1], c='red', s=1)
# plt.legend([p1, p2, p3], ['map_1', 'map_2', 'alignment'], loc='lower right', scatterpoints=1)
plt.show()
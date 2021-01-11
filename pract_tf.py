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

def extract_layer(map_input=None, bound=None, layer_type="white"):
    

    if bound == None:
        return print("Argument Error: Need a Parameter(bound) for the layer map.")

    if layer_type == "white":
        target_data = 254
        print("extract white layer")
    elif layer_type == "black":
        target_data = 0
    else:
        return print("Argument Error: layer_type should be white or black.")

    r, c = np.shape(map_input)
    map_tmp = np.zeros((5*bound, 5*bound))
    map_layer = np.zeros((r,c))

    target_data_pos = np.where(map_input==target_data)
    map_layer[target_data_pos] = 254

    map_tmp[2*bound:2*bound+r, 2*bound:2*bound+c] = map_layer

    return map_tmp


if __name__ == "__main__":
    
    img1 = cv2.imread('map4.1.png')
    img2 = cv2.imread('map4.2.png')
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    r,c = np.shape(img_gray)
    bound = max(r,c)
    
    map_white_layer = extract_layer(map_input=img1_gray, bound=bound, layer_type="white")
    map_black_layer = extract_layer(map_input=img1_gray, bound=bound, layer_type="black")

    # img_tmp = np.zeros((5*bound, 5*bound))
    # img_tmp[2*bound:2*bound+r, 2*bound:2*bound+c] = img_gray

    # img_rot = rotate(image=img_tmp, angle=0, center=(2*bound,2*bound), scale=1)
    # img_rot1 = rotate(image=img_tmp, angle=45, center=(2*bound,2*bound), scale=1)
    # plt.figure(), plt.imshow(img_rot, cmap='gray'), plt.scatter([2*bound],[2*bound], s=10, c='red')
    # plt.figure(), plt.imshow(img_rot1, cmap='gray'), plt.scatter([2*bound],[2*bound], s=10, c='red')

    map_white_layer_rot = rotate(image=map_white_layer, angle=45, center=(2*bound,2*bound), scale=1)
    map_black_layer_rot = rotate(image=map_black_layer, angle=45, center=(2*bound,2*bound), scale=1)


    # plt.figure()
    # plt.subplot(1,3,1), plt.imshow(img_gray, cmap='gray'), plt.title("Orginal")
    # plt.subplot(1,3,2), plt.imshow(map_white_layer, cmap='gray'), plt.title("White Layer")
    # plt.subplot(1,3,3), plt.imshow(map_black_layer, cmap='gray'), plt.title("Black Layer")

    # plt.figure()
    # plt.subplot(1,2,1), plt.imshow(map_white_layer_rot, cmap='gray'), plt.title("White Layer Rot")
    # plt.subplot(1,2,2), plt.imshow(map_black_layer_rot, cmap='gray'), plt.title("Black Layer Rot")
    # plt.show()

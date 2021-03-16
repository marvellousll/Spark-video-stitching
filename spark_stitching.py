from numpy import matrixlib
import pandas
import pyspark
from pyspark.sql import SparkSession
from subprocess import call
from PIL import Image
import cv2
import numpy as np
import argparse
import sys
import io
import datetime
import os

'''
sys.argv[1] = folder in which each frame is stored as a sub-folder with 6 images
sys.argv[2] = feature_extraction_method
sys.argv[3] = matcher_method
sys.argv[4] = k_val
sys.argv[5] = ratio
sys.argv[6] = output folder
'''
# def load_images_from_folder(folder):
#     images = []
#     for filename in os.listdir(folder):
#         img = cv2.imread(os.path.join(folder,filename))
#         if img is not None:
#             images.append(img)
#     return images
def getColorImage(image):
    name, img = image
    image_img = Image.open(io.BytesIO(img))
    np_img = cv2.cvtColor(np.asarray(image_img),cv2.COLOR_RGB2BGR) 
    return np_img

def getGrayscaleImage(image):
    name, img = image
    image_img = Image.open(io.BytesIO(img)).convert('L')
    np_img = np.array(image_img)
    return np_img


def getKeypointsAndDescriptors(image, feature_extraction_method):
    if feature_extraction_method == 'sift':
        descriptor = cv2.SIFT_create()

    (keypoints, descriptors) = descriptor.detectAndCompute(image, None)
    return list([[(p.pt[0], p.pt[1]) for p in keypoints], descriptors.tolist()])


def createMatcher(matcher_method, feature_extraction_method):    
    if matcher_method == "flann":
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    if matcher_method == "bruteforce":
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False) # Use cv2.NORM_HAMMING for ORB and BRISK

    return matcher


def matchFeatures2(desc1, desc2, k_val, ratio):
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False)
    matches = matcher.knnMatch(np.array([np.array([np.float32(y) for y in x]) for x in desc2]), np.array([np.array([np.float32(y) for y in x]) for x in desc1]), k=k_val)
    good_matches = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good_matches.append(m)
    return good_matches

def matchFeatures(desc1, desc2, k_val, ratio):
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False)
    matches = matcher.knnMatch(np.array([np.array([np.float32(y) for y in x]) for x in desc2]), np.array([np.array([np.float32(y) for y in x]) for x in desc1]), k=k_val)
    good_matches = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good_matches.append([m.queryIdx, m.trainIdx])
    return good_matches

# isToTheLeft counts the number of matches on the left side of image 1 and the number of matches
# on the right side of image 2. Then, it calculates the ratio of these matches to the total number
# of matches and returns this value. 
def isToTheLeft(matches, keypoints1, keypoints2, img1width, img2width):
    src_pts = np.float32([ keypoints2[m[0]] for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints1[m[1]] for m in matches ]).reshape(-1,1,2)

    num_points_for_left = 0
    for point in src_pts:
        if point[0][0] > (img2width / 2):
            num_points_for_left = num_points_for_left + 1

    for point in dst_pts:
        if point[0][0] < (img1width / 2):
            num_points_for_left = num_points_for_left + 1

    ratio_points_for_left = num_points_for_left / (len(src_pts) + len(dst_pts))
    return ratio_points_for_left > 0.5

def stitchMultImages(img_color, img_gray, img_keypoints, img_descriptors, rdd_matrix, k_val, ratio):
    num_imgs = len(img_color)
    match_matrix = [[[] for i in range(num_imgs)] for j in range(num_imgs)] #stores matches for each pair of images
    num_match_matrix = [[0 for i in range(num_imgs)] for j in range(num_imgs)] #stores number of matches for each pair of images
    print("time5 ", datetime.datetime.now()) 

    # METHOD 1: NON-DISTRIBUTED

    # for i, desc_i in enumerate(img_descriptors):
    #     for j, desc_j in enumerate(img_descriptors):
    #         if i < j:
    #             match_matrix[i][j] = matchFeatures2(desc_i, desc_j, k_val, ratio)
    #             num_match_matrix[i][j] = len(match_matrix[i][j])
    #             num_match_matrix[j][i] = num_match_matrix[i][j]
    
    # print("time8 ", datetime.datetime.now()) 
    # print(num_match_matrix)

    # METHOD 2: DISTRIBUTED WITH INDEX
    
    matrix = rdd_matrix.map(lambda d : matchFeatures(img_descriptors[d[0]], img_descriptors[d[1]], k_val, ratio)).collect()
    print("time6 ", datetime.datetime.now()) 

    k = 0
    for i in range(num_imgs):
        for j in range(num_imgs):
            if i < j:
                match_matrix[i][j] = matrix[k]
                num_match_matrix[i][j] = len(match_matrix[i][j])
                num_match_matrix[j][i] = num_match_matrix[i][j]
                k += 1

    print(num_match_matrix)
    print("time7 ", datetime.datetime.now()) 

    # Find the image that is most likely to be the edge. This code works on the principle that 
    # each non-edge image will have two other images with which it will have the most matches.
    # An edge image will only have one image that it matches well with. So, we are assuming that 
    # the image with the second highest matches that is the lowest out of all the second highest 
    # matches for all the other images, is most likely to be an edge image. This has been tested 
    # on multiple sets of images sent in random orders. 
    
    lowest_second_max = 10000000000000000000 # arbitary large number
    edge_index = -1
    neighbors = [] # stores the index of the neighboring images (i.e. images with first and second highest number of matches)
    for i in range(num_imgs):
        # get second highest matches for the image i
        first_max = max(num_match_matrix[i])
        second_max = max([x for x in num_match_matrix[i] if x != first_max ])
        neighbors.append([num_match_matrix[i].index(first_max), num_match_matrix[i].index(second_max)])
        if second_max < lowest_second_max:
            lowest_second_max = second_max
            edge_index = i

    # start with the edge image and find the next image to stitch and append it to the list. 
    # We find the image by picking one of the two neighbors that has not been added to the list previously.
    # Then move on to that image and find the next image it matches best with. Continue
    # this till all the images have been added to the list.     

    final_order = []
    neigh_index = neighbors[edge_index][0] # The edge only has one neighbor
    final_order.append(edge_index)
    final_order.append(neigh_index)
    prev = edge_index
    cur = neigh_index

    while len(final_order) != num_imgs:
        next = neighbors[cur][0] if neighbors[cur][0] != prev else neighbors[cur][1]
        final_order.append(next)
        prev = cur
        cur = next

    # now we have an ordering from one edge to the other
    # but we don't know if the ordering is from left to right or from right to left
    # we can check this by checking if the edge is on the left or right of its neighbor
    # and reverse the ordering if needed
    rev_flag = False
    if edge_index < neigh_index:
        rev_flag = isToTheLeft(match_matrix[edge_index][neigh_index], img_keypoints[edge_index], img_keypoints[neigh_index], len(img_gray[edge_index][0]), len(img_gray[neigh_index][0]))
    else:
        rev_flag = not isToTheLeft(match_matrix[neigh_index][edge_index], img_keypoints[neigh_index], img_keypoints[edge_index], len(img_gray[neigh_index][0]), len(img_gray[edge_index][0]))
    if rev_flag:
        final_order.reverse()

    print(final_order)
    print("time7 ", datetime.datetime.now()) 

    homographies = findHomography(final_order, match_matrix, img_keypoints)
    print("time8 ", datetime.datetime.now()) 

    stitched_img = stitchImages(final_order, img_color, homographies)
    print("time9 ", datetime.datetime.now()) 
    return stitched_img

# return a list of homographies
# homographies[i] is the homography from img[i+1] to img[i]
def findHomography(order, match_matrix, img_keypoints):
    num_imgs = len(img_keypoints)
    homographies = []
    for i in range(num_imgs - 1):
        index1 = order[i]
        index2 = order[i+1]

        if (index1 < index2):
            src_pts = np.float32([ img_keypoints[index2][m[0]] for m in match_matrix[index1][index2] ]).reshape(-1,1,2)
            dst_pts = np.float32([ img_keypoints[index1][m[1]] for m in match_matrix[index1][index2] ]).reshape(-1,1,2)
            homography,_ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            homographies.append(homography)
        else:
            src_pts = np.float32([ img_keypoints[index2][m[1]] for m in match_matrix[index2][index1] ]).reshape(-1,1,2)
            dst_pts = np.float32([ img_keypoints[index1][m[0]] for m in match_matrix[index2][index1] ]).reshape(-1,1,2)
            homography,_ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            homographies.append(homography)
    return homographies

# stitch images given the ordering of images and the homographies between neighboring images
# the idea is to work from the last image by stitching img[n-1] to img[n-2]
# and then stitch the stitched image to img[n-3] and so on
def stitchImages(order, img_color, homographies):
    num_imgs = len(img_color)
    stitched_img = img_color[order[-1]]
    for i in reversed(range(num_imgs - 1)):
        next_img = img_color[order[i]]
        stitched_img = cv2.warpPerspective(stitched_img, homographies[i], (len(next_img[0]) + len(stitched_img[0]), len(next_img)))
        stitched_img[0 : len(next_img), 0 : len(next_img[0])] = next_img

    return stitched_img


if __name__ == '__main__':
    sc = pyspark.SparkContext()
    spark = SparkSession(sc)

    print("time1 ", datetime.datetime.now()) 
    frames = sc.binaryFiles(sys.argv[1])
     
    rdd_frames_gray = frames.map(lambda img: getGrayscaleImage(img)).cache()
    frames_gray = rdd_frames_gray.collect()
    frames_color = frames.map(lambda img: getColorImage(img)).collect()

    num_imgs = len(frames_color)
    print("time2 ", datetime.datetime.now()) 

    # each value in frame_key_desc is a list with [keypoints, descriptors]
    key_desc = rdd_frames_gray.map(lambda img: getKeypointsAndDescriptors(img, "sift")).collect()
    img_keypoints = [x[0] for x in key_desc]
    img_descriptors = [x[1] for x in key_desc]
    print("time3 ", datetime.datetime.now())

    # create and parallelize neighboring relations into rdd
    # used to calculate match_matrix in stitchMultImages
    matrix = []
    for i, desc_i in enumerate(img_descriptors):
        for j, desc_j in enumerate(img_descriptors):
            if i < j:
               matrix.append([i, j])

    rdd_matrix = sc.parallelize(matrix, numSlices=len(matrix))
    print("time4 ", datetime.datetime.now())


    stitched_img = stitchMultImages(frames_color, frames_gray, img_keypoints, img_descriptors, rdd_matrix, int(sys.argv[4]), float(sys.argv[5]))
    cv2.imwrite("output.jpg", stitched_img)
    call(["gsutil","cp",'output.jpg', sys.argv[6]])
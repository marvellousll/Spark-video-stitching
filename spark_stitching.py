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
        descriptor = cv2.SIFT_create(nfeatures=500)
        (keypoints, descriptors) = descriptor.detectAndCompute(image, None)

    if feature_extraction_method == 'orb':
        descriptor = cv2.ORB_create()
        (keypoints, descriptors) = descriptor.detectAndCompute(image, None)

    if feature_extraction_method == 'fastbrief':
        detector = cv2.FastFeatureDetector_create()
        keypoints = detector.detect(image, None)
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        _, descriptors = brief.compute(image, keypoints)
    
    if feature_extraction_method == 'fastbrisk':
        detector = cv2.FastFeatureDetector_create()
        keypoints = detector.detect(image, None)
        br = cv2.BRISK_create()
        _, descriptors = br.compute(image, keypoints)

    if feature_extraction_method == 'starbrief':
        detector = cv2.xfeatures2d.StarDetector_create()
        keypoints = detector.detect(image, None)
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        _, descriptors = brief.compute(image, keypoints)

    if feature_extraction_method == 'brisk':
        descriptor = cv2.BRISK_create()
        (keypoints, descriptors) = descriptor.detectAndCompute(image, None)

    if feature_extraction_method == 'akaze':
        descriptor = cv2.AKAZE_create()
        (keypoints, descriptors) = descriptor.detectAndCompute(image, None)

    return list([image.tolist(), [(p.pt[0], p.pt[1]) for p in keypoints], descriptors.tolist()])


def createMatcher(matcher_method, feature_extraction_method):    
    if matcher_method == "flann":
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    if matcher_method == "bruteforce":
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False) # Use cv2.NORM_HAMMING for ORB and BRISK

    return matcher


def matchFeatures(matcher, img_i_desc, img_j_desc, k_val, ratio):
    matches = matcher.knnMatch(np.array([np.array([np.float32(y) for y in x]) for x in img_j_desc]), np.array([np.array([np.float32(y) for y in x]) for x in img_i_desc]), k=k_val)
    good_matches = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good_matches.append(m)

    return good_matches

# isToTheLeft counts the number of matches on the left side of image 1 and the number of matches
# on the right side of image 2. Then, it calculates the ratio of these matches to the total number
# of matches and returns this value. 
def isToTheLeft(matches, keypoints1, keypoints2, img1width, img2width):
    src_pts = np.float32([ keypoints2[m.queryIdx] for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints1[m.trainIdx] for m in matches ]).reshape(-1,1,2)

    num_points_for_left = 0
    for point in src_pts:
        if point[0][0] > (img2width / 2):
            num_points_for_left = num_points_for_left + 1

    for point in dst_pts:
        if point[0][0] < (img1width / 2):
            num_points_for_left = num_points_for_left + 1

    ratio_points_for_left = num_points_for_left / (len(src_pts) + len(dst_pts))
    return ratio_points_for_left > 0.5

def stitchMultImages(matcher, img_color, df_frame_key_desc, num_imgs, k_val, ratio, time4):

    match_matrix = [[[] for i in range(num_imgs)] for j in range(num_imgs)] #stores matches for each pair of images
    num_match_matrix = [[0 for i in range(num_imgs)] for j in range(num_imgs)] #stores number of matches for each pair of images

    collected = df_frame_key_desc.select("img", "keyp", "desc").toPandas() # Pandas supports more optimized conversion than mapping
    img_gray = list(collected["img"])
    img_keypoints = list(collected["keyp"])
    desc = list(collected["desc"])
    
    time5 = datetime.datetime.now()
    print("time5 ", time5)
    print("diff5 ", (time5 - time4).total_seconds()) 

    for i, desc_i in enumerate(desc):
        for j, desc_j in enumerate(desc):
            if i < j:
                match_matrix[i][j] = matchFeatures(matcher, desc_i, desc_j, k_val, ratio)
                num_match_matrix[i][j] = len(match_matrix[i][j])
                num_match_matrix[j][i] = num_match_matrix[i][j]

    print(num_match_matrix)

    time6 = datetime.datetime.now()
    print("time6 ", time6)
    print("diff6 ", (time6 - time5).total_seconds()) 

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

    time7 = datetime.datetime.now()
    print("time7 ", time7)
    print("diff7 ", (time7 - time6).total_seconds()) 

    homographies = findHomography(final_order, match_matrix, img_keypoints)
    time8 = datetime.datetime.now()
    print("time8 ", time8)
    print("diff8 ", (time8 - time7).total_seconds()) 

    stitched_img = stitchImages(final_order, img_color, homographies)
    time9 = datetime.datetime.now()
    print("time9 ", time9)
    print("diff9 ", (time9 - time8).total_seconds()) 
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
            src_pts = np.float32([ img_keypoints[index2][m.queryIdx] for m in match_matrix[index1][index2] ]).reshape(-1,1,2)
            dst_pts = np.float32([ img_keypoints[index1][m.trainIdx] for m in match_matrix[index1][index2] ]).reshape(-1,1,2)
            homography,_ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            homographies.append(homography)
        else:
            src_pts = np.float32([ img_keypoints[index2][m.trainIdx] for m in match_matrix[index2][index1] ]).reshape(-1,1,2)
            dst_pts = np.float32([ img_keypoints[index1][m.queryIdx] for m in match_matrix[index2][index1] ]).reshape(-1,1,2)
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
    num_imgs = 8
    time1 = datetime.datetime.now()
    print("time1 ", time1)
    frames = sc.binaryFiles(sys.argv[1])

    frames_color = frames.map(lambda img: getColorImage(img)).collect()
    frames_gray = frames.map(lambda img: getGrayscaleImage(img))
    # each value in frame_key_desc is a list with [gray_frame, x-coordinates of keypoints, y-coordinates of keypoints, descriptors]
    time2 = datetime.datetime.now()
    print("time2 ", time2) 
    print("diff2 ", (time2 - time1).total_seconds())
    frame_key_desc = frames_gray.map(lambda img: getKeypointsAndDescriptors(img, 'starbrief')).cache()
    time3 = datetime.datetime.now()
    print("time3 ", time3) 
    print("diff3 ", (time3 - time2).total_seconds())

    df_frame_key_desc = frame_key_desc.toDF(["img", "keyp", "desc"])
    time4 = datetime.datetime.now()
    print("time4 ", time4)
    print("diff4 ", (time4 - time3).total_seconds()) 

    matcher = createMatcher(sys.argv[3], sys.argv[2])
    stitched_img = stitchMultImages(matcher, frames_color, df_frame_key_desc, num_imgs, int(sys.argv[4]), float(sys.argv[5]), time4)
    cv2.imwrite("output.jpg", stitched_img)
    call(["gsutil","cp",'output.jpg', sys.argv[6]])
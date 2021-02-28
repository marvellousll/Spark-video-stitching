import pyspark
from pyspark.sql import SparkSession
from PIL import Image
import cv2
import numpy as np
import argparse
import sys
import io
import copyreg

'''
sys.argv[1] = folder in which each frame is stored as a sub-folder with 6 images
sys.argv[2] = feature_extraction_method
sys.argv[3] = matcher_method
sys.argv[4] = k_val
sys.argv[5] = ratio
sys.argv[6] = output folder
'''

def patch_Keypoint_pickiling(self):
    # See : https://stackoverflow.com/questions/10045363/pickling-cv2-keypoint-causes-picklingerror
    def _pickle_keypoint(keypoint):
        return cv2.KeyPoint, (
            keypoint.pt[0],
            keypoint.pt[1],
            keypoint.size,
            keypoint.angle,
            keypoint.response,
            keypoint.octave,
            keypoint.class_id,
        )
    copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoint)

def getGrayscaleImage(image):
    name, img = image
    image_img = Image.open(io.BytesIO(img)).convert('L')
    np_img = np.array(image_img)
    return np_img


def getKeypointsAndDescriptors(image, feature_extraction_method):
    if feature_extraction_method == 'sift':
        descriptor = cv2.SIFT_create()

    #patch_Keypoint_pickiling(cv2.KeyPoint)
    (keypoints, descriptors) = descriptor.detectAndCompute(image, None)
    return list([image.tolist(), [x.pt[0] for x in keypoints], [x.pt[1] for x in keypoints], descriptors.tolist()])


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
    src_pts = np.float32([ keypoints2[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints1[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    num_points_for_left = 0
    for point in src_pts:
        if point[0][0] > (img2width / 2):
            num_points_for_left = num_points_for_left + 1

    for point in dst_pts:
        if point[0][0] < (img1width / 2):
            num_points_for_left = num_points_for_left + 1

    ratio_points_for_left = num_points_for_left / (len(src_pts) + len(dst_pts))
    return ratio_points_for_left > 0.5


def stitchTwoImages(matches, keypoints1, keypoints2, img1, img2):
    src_pts = np.float32([ keypoints2[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints1[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    homography,_ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    stitched_img = cv2.warpPerspective(img2, homography, ((img1.shape[1] + img2.shape[1]), img1.shape[0])) 
    stitched_img[0 : img1.shape[0], 0 : img1.shape[1]] = img1 
    cv2.imwrite('stitched.jpg', stitched_img)
    cv2.imshow("Result", stitched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return stitched_img


def stitchMultImages(matcher, img_color, df_frame_key_desc, num_imgs, k_val, ratio):
    match_matrix = [[[] for i in range(num_imgs)] for j in range(num_imgs)] #stores matches for each pair of images
    num_match_matrix = [[0 for i in range(num_imgs)] for j in range(num_imgs)] #stores number of matches for each pair of images
    desc = df_frame_key_desc.select("desc").collect()
    
    '''
    print(type(desc))
    print(len(desc))
    print("--------")
    print(type(desc[0]))
    print(len(desc[0]))
    print("--------")
    print(type(desc[0][0]))
    print(len(desc[0][0]))
    print("--------")
    print(type(desc[0][0][0]))
    print(len(desc[0][0][0]))
    print("--------")
    print(type(desc[0][0][0][0]))
    '''
    
    for i, row_i in enumerate(desc):
        for j, row_j in enumerate(desc):
            if i < j:
                match_matrix[i][j] = matchFeatures(matcher, row_i["desc"], row_j["desc"], k_val, ratio)
                num_match_matrix[i][j] = len(match_matrix[i][j])
                num_match_matrix[j][i] = num_match_matrix[i][j]

    print(num_match_matrix)

    return #TODO: Continue from here


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
        rev_flag = isToTheLeft(match_matrix[edge_index][neigh_index], img_keypoints[edge_index], img_keypoints[neigh_index], img_gray[edge_index].shape[1], img_gray[neigh_index].shape[1])
    else:
        rev_flag = not isToTheLeft(match_matrix[neigh_index][edge_index], img_keypoints[neigh_index], img_keypoints[edge_index], img_gray[neigh_index].shape[1], img_gray[edge_index].shape[1])
    if rev_flag:
        final_order.reverse()

    print(final_order)
    homographies = findHomography(match_matrix, img_keypoints)
    stitchImages(final_order, img_color, homographies)

# return a list of homographies
# homographies[i] is the homography from img[i+1] to img[i]
def findHomography(match_matrix, img_keypoints):
    num_imgs = len(img_keypoints)
    homographies = []
    for i in range(num_imgs - 1):
        src_pts = np.float32([ img_keypoints[i+1][m.queryIdx].pt for m in match_matrix[i][i+1] ]).reshape(-1,1,2)
        dst_pts = np.float32([ img_keypoints[i][m.trainIdx].pt for m in match_matrix[i][i+1] ]).reshape(-1,1,2)
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
        stitched_img = cv2.warpPerspective(stitched_img, homographies[i], (img_color[i].shape[1] + stitched_img.shape[1], img_color[i].shape[0]))
        stitched_img[0 : img_color[i].shape[0], 0 : img_color[i].shape[1]] = img_color[i]

    cv2.imwrite('stitched.jpg', stitched_img)


if __name__ == '__main__':
    sc = pyspark.SparkContext()
    spark = SparkSession(sc)
    num_imgs = 6
    frames = sc.binaryFiles(sys.argv[1])
    frames_gray = frames.map(lambda img: getGrayscaleImage(img))
    # each value in frame_key_desc is a list with [gray_frame, point 1 of keypoints, point 2 of keypoints, descriptors]
    frame_key_desc = frames_gray.map(lambda img: getKeypointsAndDescriptors(img, "sift")).cache()
    df_frame_key_desc = frame_key_desc.toDF(["img", "key1", "key2", "desc"])
    df_frame_key_desc.printSchema()

    matcher = createMatcher(sys.argv[3], sys.argv[2])
    stitchMultImages(matcher, frames, df_frame_key_desc, num_imgs, int(sys.argv[4]), float(sys.argv[5]))

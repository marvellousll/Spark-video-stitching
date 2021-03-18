from numpy import matrixlib
from pyspark import SparkContext, SparkConf
from subprocess import call
from PIL import Image
import cv2
import numpy as np
import sys
import io
import datetime


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

    return [[(p.pt[0], p.pt[1]) for p in keypoints], descriptors]


def createMatcher(matcher_method, feature_extraction_method):    
    if matcher_method == "flann":
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    if matcher_method == "bruteforce":
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False) # Use cv2.NORM_HAMMING for ORB and BRISK

    return matcher


def matchFeatures(desc1, desc2, k_val, ratio):
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False)
    matches = matcher.knnMatch(desc2, desc1, k=k_val)
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

def stitchMultImages(img_color, img_keypoints, img_descriptors, rdd_matrix, k_val, ratio):
    num_imgs = len(img_color)
    match_matrix = [[[] for i in range(num_imgs)] for j in range(num_imgs)] #stores matches for each pair of images
    num_match_matrix = [[0 for i in range(num_imgs)] for j in range(num_imgs)] #stores number of matches for each pair of images

    matrix = rdd_matrix.map(lambda d : matchFeatures(img_descriptors[d[0]], img_descriptors[d[1]], k_val, ratio)).collect()

    k = 0
    for i in range(num_imgs):
        for j in range(num_imgs):
            if i < j:
                match_matrix[i][j] = matrix[k]
                num_match_matrix[i][j] = len(match_matrix[i][j])
                num_match_matrix[j][i] = num_match_matrix[i][j]
                k += 1

    print("time3 ", datetime.datetime.now()) 

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
        rev_flag = isToTheLeft(match_matrix[edge_index][neigh_index], img_keypoints[edge_index], img_keypoints[neigh_index], len(img_color[edge_index][0]), len(img_color[neigh_index][0]))
    else:
        rev_flag = not isToTheLeft(match_matrix[neigh_index][edge_index], img_keypoints[neigh_index], img_keypoints[edge_index], len(img_color[neigh_index][0]), len(img_color[edge_index][0]))
    if rev_flag:
        final_order.reverse()

    #print(final_order)
    homographies = findHomography(final_order, match_matrix, img_keypoints)
    stitched_img = stitchImages(final_order, img_color, homographies)

    return stitched_img

def stitchMultImagesWithOrdering(img_color, img_keypoints, img_descriptors, k_val, ratio):
    num_imgs = len(img_color)
    match_matrix = [[[] for i in range(num_imgs)] for j in range(num_imgs)] #stores matches for each pair of images

    final_order = range(num_imgs)    
    for i in range(num_imgs - 1):
        match_matrix[i][i+1] = matchFeatures(img_descriptors[i], img_descriptors[i+1], k_val, ratio)
    
    homographies = findHomography(final_order, match_matrix, img_keypoints)
    stitched_img = stitchImages(final_order, img_color, homographies)

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

def stitchTask(sc, frame_index, input_dir, feature_extraction_method, k_val, ratio, output_dir):
    print("start task " + frame_index + " " + str(datetime.datetime.now())) 
    rdd_frames = sc.binaryFiles(input_dir + frame_index)
    frames_color = rdd_frames.map(lambda img: getColorImage(img)).collect()

    print("time1 ", datetime.datetime.now()) 
    key_desc = rdd_frames.map(lambda img: getGrayscaleImage(img)).map(lambda img: getKeypointsAndDescriptors(img, feature_extraction_method)).collect()
    img_keypoints = [x[0] for x in key_desc]
    img_descriptors = [x[1] for x in key_desc]
    print("time2 ", datetime.datetime.now()) 

    # create and parallelize neighboring relations into rdd
    # used to calculate match_matrix in stitchMultImages
    matrix = []
    num_imgs = len(frames_color)
    for i in range(num_imgs):
        for j in range(num_imgs):
            if i < j:
               matrix.append([i, j])

    rdd_matrix = sc.parallelize(matrix, numSlices=len(matrix))

    stitched_img = stitchMultImages(frames_color, img_keypoints, img_descriptors, rdd_matrix, k_val, ratio)
    print("time4 ", datetime.datetime.now()) 

    cv2.imwrite("output" + frame_index + ".jpg", stitched_img)
    call(["gsutil","cp","output" + frame_index + ".jpg", output_dir])
    print("end task " + frame_index + " " + str(datetime.datetime.now()))

def stitchWithOrderingTask(sc, frame_index, input_dir, feature_extraction_method, k_val, ratio, output_dir):
    print("start task " + frame_index + " " + str(datetime.datetime.now())) 
    rdd_frames = sc.binaryFiles(input_dir + frame_index)

    frames_color = rdd_frames.map(lambda img: getColorImage(img)).collect()
    key_desc = rdd_frames.map(lambda img: getGrayscaleImage(img)).map(lambda img: getKeypointsAndDescriptors(img, feature_extraction_method)).collect()

    img_keypoints = [x[0] for x in key_desc]
    img_descriptors = [x[1] for x in key_desc]

    stitched_img = stitchMultImagesWithOrdering(frames_color, img_keypoints, img_descriptors, k_val, ratio)
    print("time4 ", datetime.datetime.now()) 
    cv2.imwrite("output" + frame_index + ".jpg", stitched_img)
    call(["gsutil","cp","output" + frame_index + ".jpg", output_dir])
    print("end task " + frame_index + " " + str(datetime.datetime.now()))

if __name__ == '__main__':
    conf = SparkConf()
    conf.set('spark.scheduler.mode', 'FAIR')
    sc = SparkContext(conf=conf)

    num_frames = int(sys.argv[1])
    input_dir = sys.argv[2]
    feature_extraction_method = sys.argv[3]
    matcher_method = sys.argv[4]
    k_val = int(sys.argv[5])
    ratio = float(sys.argv[6])
    output_dir = sys.argv[7]

    # when stitching images in input/frame2, set the second argument as str(1)
    stitchTask(sc, str(10), input_dir, feature_extraction_method, k_val, ratio, output_dir)
    # stitchTask(sc, str(2), input_dir, feature_extraction_method, k_val, ratio, output_dir)
   
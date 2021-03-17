import cv2
import numpy as np
import argparse

def getGrayscaleImage(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def getColorImage(path):
    return cv2.imread(path)

def getKeypointsAndDescriptors(image, feature_extraction_method):
    if feature_extraction_method == 'sift':
        print("Using SIFT")
        descriptor = cv2.SIFT_create()
        (keypoints, descriptors) = descriptor.detectAndCompute(image, None)

    if feature_extraction_method == 'orb':
        print("Using ORB")
        descriptor = cv2.ORB_create()
        (keypoints, descriptors) = descriptor.detectAndCompute(image, None)

    if feature_extraction_method == 'fastbrief':
        print("Using FAST_BRIEF")
        detector = cv2.FastFeatureDetector_create()
        keypoints = detector.detect(image, None)
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        _, descriptors = brief.compute(image, keypoints)
    
    if feature_extraction_method == 'fastbrisk':
        print("Using FAST_BRISK")
        detector = cv2.FastFeatureDetector_create()
        keypoints = detector.detect(image, None)
        br = cv2.BRISK_create()
        _, descriptors = br.compute(image, keypoints)

    if feature_extraction_method == 'starbrief':
        print("Using STAR_BRIEF")
        detector = cv2.xfeatures2d.StarDetector_create()
        keypoints = detector.detect(image, None)
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        _, descriptors = brief.compute(image, keypoints)

    if feature_extraction_method == 'brisk':
        print("Using BRISK")
        descriptor = cv2.BRISK_create()
        (keypoints, descriptors) = descriptor.detectAndCompute(image, None)

    if feature_extraction_method == 'akaze':
        print("Using AKAZE")
        descriptor = cv2.AKAZE_create()
        (keypoints, descriptors) = descriptor.detectAndCompute(image, None)

    if feature_extraction_method == 'kaze':
        print("Using KAZE")
        descriptor = cv2.KAZE_create()
        (keypoints, descriptors) = descriptor.detectAndCompute(image, None)

    
    return (keypoints, descriptors)

def createMatcher(matcher_method, feature_extraction_method):    
    if matcher_method == "flann":
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    if matcher_method == "bruteforce":
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False) # Use cv2.NORM_HAMMING for ORB and BRISK

    return matcher

def matchFeatures(matcher, img1, img2, keypoints1, descriptors1, keypoints2, descriptors2, k_val, ratio):
    matches = matcher.knnMatch(descriptors2, descriptors1, k=k_val)
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


def stitchMultImages(matcher, img_color, img_gray, img_keypoints, img_descriptors, k_val, ratio):
    num_imgs = len(img_gray)
    match_matrix = [[[] for i in range(num_imgs)] for j in range(num_imgs)] #stores matches for each pair of images
    num_match_matrix = [[0 for i in range(num_imgs)] for j in range(num_imgs)] #stores number of matches for each pair of images
    for i in range(num_imgs):
        j = i + 1
        while j < num_imgs:
            # get matches for img i and img j
            match_matrix[i][j] = matchFeatures(matcher, img_gray[i], img_gray[j], img_keypoints[i], img_descriptors[i], img_keypoints[j], img_descriptors[j], k_val, ratio)
            num_match_matrix[i][j] = len(match_matrix[i][j])
            num_match_matrix[j][i] = len(match_matrix[i][j])
            j = j + 1

    print("NUM MATCH MATRIX - ", num_match_matrix)

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
    homographies = findHomography(final_order, match_matrix, img_keypoints)
    stitchImages(final_order, img_color, homographies)

# return a list of homographies
# homographies[i] is the homography from img[i+1] to img[i]
def findHomography(order, match_matrix, img_keypoints):
    num_imgs = len(img_keypoints)
    homographies = []
    for i in range(num_imgs - 1):
        index1 = order[i]
        index2 = order[i+1]

        if (index1 < index2):
            src_pts = np.float32([ img_keypoints[index2][m.queryIdx].pt for m in match_matrix[index1][index2] ]).reshape(-1,1,2)
            dst_pts = np.float32([ img_keypoints[index1][m.trainIdx].pt for m in match_matrix[index1][index2] ]).reshape(-1,1,2)
            homography,_ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            homographies.append(homography)
        else:
            src_pts = np.float32([ img_keypoints[index2][m.trainIdx].pt for m in match_matrix[index2][index1] ]).reshape(-1,1,2)
            dst_pts = np.float32([ img_keypoints[index1][m.queryIdx].pt for m in match_matrix[index2][index1] ]).reshape(-1,1,2)
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
        stitched_img = cv2.warpPerspective(stitched_img, homographies[i], (next_img.shape[1] + stitched_img.shape[1], next_img.shape[0]))
        stitched_img[0 : next_img.shape[0], 0 : next_img.shape[1]] = next_img

    cv2.imwrite('stitched.jpg', stitched_img)

def main(path1, path2, path3, path4, path5, path6, path7, path8, feature_extraction_method, matcher_method, k_val, ratio):
    path_list = [path1, path2, path3, path4, path5, path6, path7, path8]
    img_color = []
    img_gray = []
    img_keypoints = []
    img_descriptors = []
    for img_num, path in enumerate(path_list):
        img_color.append(getColorImage(path))
        img_gray.append(getGrayscaleImage(path))
        (keypoints, descriptors) = getKeypointsAndDescriptors(img_gray[img_num], feature_extraction_method)
        img_keypoints.append(keypoints)
        img_descriptors.append(descriptors)

    matcher = createMatcher(matcher_method, feature_extraction_method)
    stitchMultImages(matcher, img_color, img_gray, img_keypoints, img_descriptors, k_val, ratio)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stitch Two Images Together')
    parser.add_argument('--image1', required=True,
                        help='the path to image 1')
    parser.add_argument('--image2', required=True,
                        help='the path to image 2')
    parser.add_argument('--image3', required=True,
                        help='the path to image 3')
    parser.add_argument('--image4', required=True,
                        help='the path to image 4')   
    parser.add_argument('--image5', required=True,
                        help='the path to image 5')
    parser.add_argument('--image6', required=True,
                        help='the path to image 6') 
    parser.add_argument('--image7', required=True,
                        help='the path to image 7') 
    parser.add_argument('--image8', required=True,
                        help='the path to image 8')                  
    parser.add_argument('--fmethod', required=True,
                        help='feature extraction method')
    parser.add_argument('--mmethod', required=True,
                        help='matcher method')
    parser.add_argument('--k', required=True,
                        help='k value for matcher')
    parser.add_argument('--ratio', required=True,
                        help='ratio for good match distance')
    args = parser.parse_args()
    main(path1=args.image1, path2=args.image2, path3=args.image3, path4=args.image4, path5=args.image5, path6=args.image6, path7=args.image7, path8=args.image8, feature_extraction_method=args.fmethod, matcher_method=args.mmethod, k_val=int(args.k), ratio=float(args.ratio))



import cv2
import numpy as np
import argparse

def getGrayscaleImage(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def getColorImage(path):
    return cv2.imread(path)

def getKeypointsAndDescriptors(image, feature_extraction_method):
    if feature_extraction_method == 'sift':
        descriptor = cv2.SIFT_create()
    #TODO: Implement SURF, ORB, etc.
    
    return descriptor.detectAndCompute(image, None)

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
    return ratio_points_for_left


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


def stitchMultImages(matcher, img_gray, img_keypoints, img_descriptors, k_val, ratio):
    match_matrix = [[0 for i in range(len(img_gray))] for j in range(len(img_gray))] #stores matches for each pair of images
    num_match_matrix = [[0 for i in range(len(img_gray))] for j in range(len(img_gray))] #stores number of matches for each pair of images
    for i,_ in enumerate(img_gray):
        match_matrix[i][i] = [] 
        num_match_matrix[i][i] = 0
        j = i+1
        while j < len(img_gray):
            # get matches for img i and img j
            match_matrix[i][j] = matchFeatures(matcher, img_gray[i], img_gray[j], img_keypoints[i], img_descriptors[i], img_keypoints[j], img_descriptors[j], k_val, ratio)
            match_matrix[j][i] = [] # no need to waste memory by storing these matches
            num_match_matrix[i][j] = len(match_matrix[i][j])
            num_match_matrix[j][i] = len(match_matrix[i][j])
            j = j + 1

    # Find the image that is most likely to be the edge. This code works on the principle that 
    # each non-edge image will have two other images with which it will have the most matches.
    # An edge image will only have one image that it matches well with. So, we are assuming that 
    # the image with the second highest matches that is the lowest out of all the second highest 
    # matches for all the other images, is most likely to be an edge image. This has been tested 
    # on multiple sets of images sent in random orders. 
     
    edge_img_second_max = max(num_match_matrix[0])
    edge_img_num = 0
    for i in range(len(img_gray)):
        # get second highest matches for the image i
        second_max = max([x for x in num_match_matrix[i] if x != max(num_match_matrix[i])])
        if second_max < edge_img_second_max:
            edge_img_second_max = second_max
            edge_img_num = i

    # start with the edge image and find the image that it matches best with and append it to the 
    # list for now. Then move on to that image and find the next image it matches best with. Continue
    # this till all the images have been added to the list. 
    final_order = []
    final_order.append(edge_img_num)
    i = edge_img_num
    ratios_for_left = 0
    while True:
        included_matches = [num_match_matrix[i][j] for j,_ in enumerate(num_match_matrix[i]) if j not in final_order]
        max_matches = max(included_matches)
        adjacent_img_num = num_match_matrix[i].index(max_matches)

        # The isToTheLeft function returns the ratio of matches that indicate that the adjacent image is
        # to the left of image i
        if i < adjacent_img_num:
            ratios_for_left = ratios_for_left + isToTheLeft(match_matrix[i][adjacent_img_num], img_keypoints[i], img_keypoints[adjacent_img_num], img_gray[i].shape[1], img_gray[adjacent_img_num].shape[1])
        else:
            ratios_for_left = ratios_for_left + (1 - isToTheLeft(match_matrix[adjacent_img_num][i], img_keypoints[adjacent_img_num], img_keypoints[i], img_gray[adjacent_img_num].shape[1], img_gray[i].shape[1]))

        final_order.append(adjacent_img_num)

        if len(final_order) == len(img_gray):
            break

        i = adjacent_img_num
    
    # if overall, the location matches indicate that the adjacent images should be to the left 
    # of the starting edge image, the order is reversed.
    if (ratios_for_left / (len(img_gray) - 1)) > 0.5:
        final_order.reverse()

    print(final_order)
    return final_order


def main(path1, path2, path3, path4, path5, path6, feature_extraction_method, matcher_method, k_val, ratio):
    path_list = [path1, path2, path3, path4, path5, path6]
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
    return stitchMultImages(matcher, img_gray, img_keypoints, img_descriptors, k_val, ratio)


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
    parser.add_argument('--fmethod', required=True,
                        help='feature extraction method')
    parser.add_argument('--mmethod', required=True,
                        help='matcher method')
    parser.add_argument('--k', required=True,
                        help='k value for matcher')
    parser.add_argument('--ratio', required=True,
                        help='ratio for good match distance')
    args = parser.parse_args()
    main(path1=args.image1, path2=args.image2, path3=args.image3, path4=args.image4, path5=args.image5, path6=args.image6, feature_extraction_method=args.fmethod, matcher_method=args.mmethod, k_val=int(args.k), ratio=float(args.ratio))



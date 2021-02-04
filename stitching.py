import cv2
import numpy as np
import argparse

def getImage(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def getKeypointsAndDescriptors(image, feature_extraction_method):
    if feature_extraction_method == 'sift':
        descriptor = cv2.SIFT_create()
    # TODO: Get yolo descriptor
    #elif feature_extraction_method == 'yolo':
    
    (keypoints, descriptors) = descriptor.detectAndCompute(image, None)
    image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Image with Keypoints", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    return (keypoints, descriptors)

def createMatcher(matcher_method, feature_extraction_method):    
    
    # TODO: How to match YOLO feature descriptors?
    #if feature_extraction_method == "yolo":

    #elif: 

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
    good_matches_for_draw = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good_matches.append(m)
            good_matches_for_draw.append([m])

    image_with_matches = cv2.drawMatchesKnn(img2, keypoints2, img1, keypoints1, good_matches_for_draw, None, flags = 2)
    cv2.imshow("Image with Good Matches", image_with_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return good_matches

def stitchImages(matches, keypoints1, keypoints2, img1, img2):
    src_pts = np.float32([ keypoints2[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints1[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    homography,_ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    stitched_img = cv2.warpPerspective(img2, homography, ((img1.shape[1] + img2.shape[1]), img1.shape[0])) 
    stitched_img[0 : img1.shape[0], 0 : img1.shape[1]] = img1 
    cv2.imwrite('stitched.jpg', stitched_img)
    cv2.imshow("Result", stitched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(path1, path2, feature_extraction_method, matcher_method, k_val, ratio):
    img1 = getImage(path1)
    img2 = getImage(path2)

    (keypoints1, descriptors1) = getKeypointsAndDescriptors(img1, feature_extraction_method)
    (keypoints2, descriptors2) = getKeypointsAndDescriptors(img2, feature_extraction_method)

    matcher = createMatcher(matcher_method, feature_extraction_method)

    matches = matchFeatures(matcher, img1, img2, keypoints1, descriptors1, keypoints2, descriptors2, k_val, ratio)
    stitchImages(matches, keypoints1, keypoints2, img1, img2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stitch Two Images Together')
    parser.add_argument('--image1', required=True,
                        help='the path to image 1')
    parser.add_argument('--image2', required=True,
                        help='the path to image 2')
    parser.add_argument('--fmethod', required=True,
                        help='feature extraction method')
    parser.add_argument('--mmethod', required=True,
                        help='matcher method')
    parser.add_argument('--k', required=True,
                        help='k value for matcher')
    parser.add_argument('--ratio', required=True,
                        help='ratio for good match distance')
    args = parser.parse_args()
    main(path1=args.image1, path2=args.image2, feature_extraction_method=args.fmethod, matcher_method=args.mmethod, k_val=int(args.k), ratio=float(args.ratio))



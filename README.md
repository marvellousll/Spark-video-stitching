The arguments for stitching.py are as follows:

1. --image1 = path to image 1
2. --image2 = path to image 2
3. --fmethod = feature extraction method
4. --mmethod = feature matching method
5. --k = value of k for kNN matcher
6. --ratio = ratio to tune which matches should be considered 'good'

sample command to run:

python3 stitching.py --image1 ./img1.png --image2 ./img2.png --fmethod sift --mmethod bruteforce --k 2 --ratio 0.75

TODO:
1. Stitch multiple images together
2. Compare SIFT, ORB and SURF for speed and accuracy (SURF with Upright flag is supposed to have a good result)
3. Compare BruteForce and Flann as Feature Matching algorithms in a Spark environment (fine tune Flann Matcher to improve efficiency without losing too much accuracy)
4. Compare using ratio testing vs crossCheck for BruteForce Matcher (no crossCheck for Flann) 
5. Use laplacian bending or other such methods to remove seam from stitched image (in general improve the final result)

Note: Using opencv-python version 4.5.1

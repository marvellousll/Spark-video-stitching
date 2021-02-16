The arguments for stitching.py are as follows:

1. --image1 = path to image 1
2. --image2 = path to image 2
3. --image1 = path to image 3
4. --image2 = path to image 4
5. --image1 = path to image 5
6. --image2 = path to image 6
7. --fmethod = feature extraction method
8. --mmethod = feature matching method
9. --k = value of k for kNN matcher
10. --ratio = ratio to tune which matches should be considered 'good'

sample command to run:

python3 stitching.py --image1 ./img6.png --image2 ./img2.png --image3 ./img5.png --image4 ./img1.png --image5 ./img4.png --image6 ./img3.png --fmethod sift --mmethod bruteforce --k 2 --ratio 0.75

TODO:
1. Stitch multiple images together
2. Compare SIFT, ORB and SURF for speed and accuracy (SURF with Upright flag is supposed to have a good result)
3. Compare BruteForce and Flann as Feature Matching algorithms in a Spark environment (fine tune Flann Matcher to improve efficiency without losing too much accuracy)
4. Compare using ratio testing vs crossCheck for BruteForce Matcher (no crossCheck for Flann) 
5. Use laplacian bending or other such methods to remove seam from stitched image (in general improve the final result)

Note: Using opencv-python version 4.5.1

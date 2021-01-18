# CS219-W21-Spark
Here is the repo for CS219 Winter 2021. **Please open a new branch for your team first and develop on it later.** You can find some related materials here: https://docs.google.com/document/d/1rNzr85FbqccfVWHc-urn5GvvPYqEfFTtRjqnQr1DJlM/edit?usp=sharing. 


## Topic 1: Cloud Anchor Extraction with Spark
To enable the collaborations between multiple users’ views for AR applications, feature points need to be extracted from objects in each camera stream. If users share some common objects in the views, these objects can be the anchor to establish the connection between real-world positions. However, extracting the feature points with current algorithms like SIFT is computation-intensive. This project explores the server cluster to speed up the processing with Spark, including: 
- Decoupling the procedures in the feature extraction to build the distributive processing with Spark
- Combining the feature extraction algorithm like SIFT with ML models like Yolo to improve the accuracy
- Comparing the performance between the single-server solution and the distributed solution

## Topic 3: Distributed OpenPose with Spark 
OpenPose provides video analytics for body, foot, face, and hands estimations. Due to the processing overhead, performing the human skeleton analytics with OpenPose on large-scale videos is still challenging. The server cluster in the cloud provides new opportunities and source videos can be split into frames, which can be processed distributively. This project explores to leverage the server cluster to speed up the OpenPose processing with Spark, including: 
- Develop the distributed processing system with Spark for video analytics so that the video frames can be split into several server nodes for further processing
- Comparing the OpenPose performance between the single-server solution and the distributed solution
reference: https://github.com/CMU-Perceptual-Computing-Lab/openpose 

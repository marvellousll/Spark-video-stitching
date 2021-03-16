# To run: python3 v2f.py arg1 arg2 arg3
# arg1: path of the source video 
# arg2: path for the frames to be stored
# arg3: name of video

import cv2
import os
import sys


# test.mp4
# /Users/hangl/Desktop/CS219/CS219-W21-Spark/frames
video = cv2.VideoCapture(sys.argv[1])
framePath = sys.argv[2]
name = sys.argv[3]

sec = 0
frameRate = 0.05 # 20 fps
count=1

def videoToFrames(sec):
    video.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    success,image = video.read()
    if success:
    	directory = framePath + '/' + 'frame' + str(count)
    	if not os.path.exists(directory):
        	os.makedirs(directory)
    	cv2.imwrite(os.path.join(directory, name+".jpg"), image)

    return success


successCode = videoToFrames(sec)

while successCode:
    count += 1
    sec = round(sec + frameRate, 2)
    successCode = videoToFrames(sec)

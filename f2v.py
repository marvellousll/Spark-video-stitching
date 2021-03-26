# To run: python3 f2v.py arg1 arg2
# arg1: path to the set of frames
# arg2: path for the generated video

import cv2
import numpy as np
import os
from os.path import isfile, join
import sys
from PIL import Image

# frames
# output.mp4
pathIn= sys.argv[1]
pathOut = sys.argv[2]

fps = 20
frame_array = []

files = [f for f in os.listdir(pathIn) if f.endswith(".jpg")]

# sort the file names properly
#files.sort(key = lambda x: x[5:-4])
files.sort(key=lambda item: (len(item), item))


frame = cv2.imread(os.path.join(pathIn, files[0]))
height, width, layers = frame.shape
size = (width,height)



for i in range(len(files)):
    filename = pathIn + '/' + files[i]
    print(filename)
    img = cv2.imread(filename)
    frame_array.append(img)
    
    
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
    
out.release()



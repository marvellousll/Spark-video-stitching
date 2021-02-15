import cv2
import numpy as np
import os
from os.path import isfile, join

# change paths here
pathIn= 'frames'
pathOut = 'output.mp4'

fps = 0.05
frame_array = []

#files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
files = [f for f in os.listdir(pathIn) if f.endswith(".jpg")]

frame = cv2.imread(os.path.join(pathIn, files[0]))
height, width, layers = frame.shape
size = (width,height)


# sort the file names properly
files.sort(key = lambda x: x[5:-4])


for i in range(len(files)):
    filename = pathIn + files[i]
    img = cv2.imread(filename)
    frame_array.append(img)
    
    
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
    
out.release()

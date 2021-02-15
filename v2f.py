import cv2
import os

# manually change path here
video = cv2.VideoCapture('test.mp4')
framePath = '/Users/hangl/Desktop/CS219/code/frames'

sec = 0
frameRate = 0.05 # 20 fps
count=1

def videoToFrames(sec):
    video.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    success,image = video.read()
    if success:
        cv2.imwrite(os.path.join(framePath, "image"+str(count)+".jpg"), image)
    return success


successCode = videoToFrames(sec)

while successCode:
    count += 1
    sec = round(sec + frameRate, 2)
    successCode = videoToFrames(sec)

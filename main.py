# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import time
import cv2
import pytesseract
from hand import *
from read import *

#global variable
read_flag = 0

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
args = vars(ap.parse_args())

# initialize the original frame dimensions, new frame dimensions,
# and ratio between the dimensions
(W, H) = (None, None)
(newW, newH) = (320, 320)
(rW, rH) = (None, None)

#grab the reference to the web cam
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

# start the FPS throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    # check to see if we have reached the end of the stream
    if frame is None:
        break

    # resize the frame, maintaining the aspect ratio
    frame = imutils.resize(frame, width=1000)
    orig = frame.copy()

    # if our frame dimensions are None, we still need to compute the
    # ratio of old frame dimensions to new frame dimensions
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        rW = W / float(newW)
        rH = H / float(newH)

    # resize the frame, this time ignoring aspect ratio
    frame = cv2.resize(frame, (newW, newH))

    # find gesture 
    hand = isHand(frame)
    print(hand)
    
    # gesture if hand is...
    if hand is 5:

        read_flag = 1
        play("stop.mp3", 24000)

    if hand is 8:

        read_flag = 0

    if hand is 2:
        
        translate('now.txt','en')
        txt2mp3('trans_en', 'en')
        play('trans_en.mp3', 24000)


    #read
    if read_flag is 0:
        read_flag = read(read_flag, orig)
    

    # update the FPS counter
    fps.update()

    # show the output frame
    cv2.imshow("Text Detection", orig)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()

# otherwise, release the file pointer
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()
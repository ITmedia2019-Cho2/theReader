# import the necessary packages
import argparse
import configparser
import cv2  
import numpy as np
import os
import pygame
import re
import time

from googletrans import Translator
from gtts import gTTS
from imutils.object_detection import non_max_suppression
from PIL import Image
from pytesseract import *


tesseract_cmd = 'tesseract'

def writeFile(text):

    
    f = open("now.txt", 'w')
    #loop over the results
    for (startX, startY, endX, endY, text) in text:

        f.write(text)
        
    f.close()

def recognize(img):

    width = 1600
    height = 1600
    padding = 0.5
    
    # load the input image and grab the image dimensions
    orig = img.copy()
    (H, W) = img.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (width, height)
    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(img, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    net = cv2.dnn.readNet('frozen_east_text_detection.pb')

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # initialize 

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.5:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # draw the bounding box on the image
    #output = orig.copy()
    #cv2.rectangle(output, (startX, startY), (endX, endY), (0, 255, 0), 2)

    results = []

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # in order to obtain a better OCR of the text we can potentially
        # apply a bit of padding surrounding the bounding box -- here we
        # are computing the deltas in both the x and y directions
        dX = int((endX - startX) * padding)
        dY = int((endY - startY) * padding)

        # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(W, endX + (dX * 2))
        endY = min(H, endY + (dY * 2))

        # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(W, endX + (dX * 2))
        endY = min(H, endY + (dY * 2))
    
        # extract the actual padded ROI
        roi = orig[startY:endY, startX:endX]

        # in order to apply Tesseract v4 to OCR text we must supply
        # (1) a language, (2) an OEM flag of 4, indicating that the we
        # wish to use the LSTM neural net model for OCR, and finally
        # (3) an OEM value, in this case, 7 which implies that we are
        # treating the ROI as a single line of text
        # in order to apply Tesseract v4 to OCR text
        config = ("-l kor --oem 1 --psm 7")
        text = pytesseract.image_to_string(roi, config=config)

        # add the bounding box coordinates and OCR'd text to the list
        # of results
        results.append((startX, startY, endX, endY, text))
  
    # sort the results bounding box coordinates from top to bottom
    results = sorted(results, key=lambda r:r[0])
    writeFile('now', results)
    txt2mp3('now', 'ko')
    
    # strip out non-ASCII text so we can draw the text on the image
    # using OpenCV, then draw the text and a bounding box surrounding
    # the text region of the input image
    #text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    

def writeFile(file, results):

    print(results)
    ff = 0

    for (startX, startY, endX, endY, text) in results:
        
        # remove except korean
        kor = re.compile('[^ \u3131-\u3163\uac00-\ud7a3]+')
        num = re.compile('[^0-9]')
        spchar = re.compile('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]')

        result = spchar.sub('', text)
        result = text.strip(num.sub('', result))
        result = kor.sub('', result)

        if ff is 0:
            f = open( file+'.txt', "w+")
            ff = 1
        if ff is 1:
            f = open( file+'.txt', "a")
        
        f.write(result)
        f.close()


def txt2mp3(file, lan):

    FLIST = open(file+'.txt', "r").read()

    print("please wait... reading")

    TTS = gTTS(text = str(FLIST), lang=lan)

    TTS.save( file+'.mp3' )


def play(music, freq):

    bitsize = -16
    channels = 1
    buffer = 2048

    pygame.mixer.init(freq, bitsize, channels, buffer)
    pygame.mixer.music.load(music)
    pygame.mixer.music.play()

    clock = pygame.time.Clock()
    while pygame.mixer.music.get_busy():
        clock.tick(30)
    pygame.mixer.quit()

    #os.system("book_s.m4a")


def translate(file, lang):

    translator = Translator()

    FLIST = open(file, "r").read()

    tr_results = translator.translate(text=str(FLIST), src='ko', dest=lang)
    
    FLIST = open('trans_'+lang+'.txt', "w")
    FLIST.write(str(tr_results.text))
    FLIST.close()


def read(flag, img):

    flag = 1
    recognize(img)
    play("now.mp3", 24000)
    '''
    #init now.txt
    a = []
    a.append((0, 0, 0, 0, '못 읽었어요'))
    writeFile(a)
    '''
    play("book_s2.wav", 44100)

    #time.sleep(1)
    flag=0

    return flag

    

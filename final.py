import cv2
import numpy as np
import math
import sys
import random

#Jhonathan Viudes 8532001

originalImg = cv2.imread(str(sys.argv[1]), cv2.IMREAD_GRAYSCALE)  #read image grayscale
outerBox =  np.empty(originalImg.shape)
width, height = originalImg.shape

processedImg = cv2.GaussianBlur(originalImg,(11,11),0)
print originalImg.shape

processedImg = cv2.adaptiveThreshold(processedImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,5,2)
processedImg = cv2.bitwise_not(processedImg)

kernel = np.matrix([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
processedImg = cv2.dilate(processedImg, kernel, iterations=1)

count = 0
maxA = -1
maxPt = (0,0)

for y in range(height):
    for x in range(width):
        if processedImg[x][y] >= 128:
            area =  cv2.floodFill(processedImg,None,(y,x), cv2.cv.RGB(0,0,64))
            if area > maxA :
                maxA = area
                maxPt = (y,x)


retval, rect = cv2.floodFill(processedImg,None,maxPt, cv2.cv.RGB(255,255,255))

for y in range(height):
    for x in range(width):
        if (processedImg[x][y] == 64) and (x!=maxPt[1]) and (y!=maxPt[0]):
            area =  cv2.floodFill(processedImg,None,(y,x), cv2.cv.RGB(0,0,0))

processedImg = cv2.erode(processedImg, kernel, iterations=1)

lines = cv2.HoughLines(processedImg,1,np.pi/180,200)

def drawLine(line,img,rgb):
    if line[1] != 0:
        m = int(-1/math.tan(line[1]))
        c = int(line[0]/math.sin(line[1]))
        cv2.line(processedImg,(0,c),(width,m*width+c),rgb)
    else:
        cv2.line(processedImg,(line[0],0),(line[0],height),rgb)
    return



for i in range(len(lines)):
    drawLine(lines[0][i],processedImg,cv2.cv.RGB(0,0,255))


# Set up the detector with default parameters.
#detector = cv2.SimpleBlobDetector()
# Detect blobs.
#keypoints = detector.detect(processedImg)
#processedImg = cv2.drawKeypoints(processedImg, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#area = cv2.floodFill(outerBox, Point(x,y), CV_RGB(0,0,0))

cv2.imshow("originalImg", originalImg)
cv2.imshow("processedImg", processedImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

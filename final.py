import cv2
import numpy as np
import math
import sys
import random
#Jhonathan Viudes 8532001

originalImg = cv2.imread(str(sys.argv[1]), cv2.IMREAD_GRAYSCALE)  #read image grayscale

processedImg = cv2.GaussianBlur(originalImg,(11,11),0)


cv.AdaptiveThreshold(processedImg, processedImg, maxValue, adaptive_method=CV_ADAPTIVE_THRESH_MEAN_C, thresholdType=CV_THRESH_BINARY, blockSize=3, param1=5)

ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

adaptiveThreshold(sudoku, outerBox, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 2);

#GaussianBlur(InputArray src, OutputArray dst, Size ksize, double sigmaX, double sigmaY=0, int borderType=BORDER_DEFAULT )
#GaussianBlur(sudoku, sudoku, Size(11,11), 0);

cv2.imshow("originalImg", originalImg)
cv2.imshow("processedImg", processedImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

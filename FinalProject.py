import cv2
import numpy as np
import math
import sys
import random
import pytesseract as tes
import Image
import re


#------------------------------- Sudoku Solver -----------------------------------

# Print the solution of the sudoku board
def print_board(arr):
    for i in range(9):
        for j in range(9):
            if(j == 3) or (j == 6):
                print '|',
            print int(arr[i][j]),
        if(i == 2) or (i == 5):
            print ('\n---------------------')
        else:
            print

# Find the next empty box
def find_empty_location(arr,l):
    for row in range(9):
        for col in range(9):
            if(arr[row][col]==0):
                l[0]=row
                l[1]=col
                return True
    return False

# Checks if 'num' already exists in the row
def used_in_row(arr,row,num):
    for i in range(9):
        if(arr[row][i] == num):
            return True
    return False
 
# Checks if 'num' already exists in the column
def used_in_col(arr,col,num):
    for i in range(9):
        if(arr[i][col] == num):
            return True
    return False
 
# Checks if 'num' already exists in the 3x3 box
def used_in_box(arr,row,col,num):
    for i in range(3):
        for j in range(3):
            if(arr[i+row][j+col] == num):
                return True
    return False
 
def check_location_is_safe(arr,row,col,num):
    # Check if 'num' is not already placed in current row,
    # current column and current 3x3 box
    return not used_in_row(arr,row,num) and not used_in_col(arr,col,num) and not used_in_box(arr,row - row%3,col - col%3,num)
 
def solve_sudoku(arr):
     
    # 'l' is a list variable that keeps the record of row and col in find_empty_location Function    
    l=[0,0]
     
    # If there is no unassigned location, we are done    
    if(not find_empty_location(arr,l)):
        return True
     
    # Assigning list values to row and col that we got from the above Function 
    row=l[0]
    col=l[1]
     
    # consider digits 1 to 9
    for num in range(1,10):
         
        # if looks promising
        if(check_location_is_safe(arr,row,col,num)):
             
            # make tentative assignment
            arr[row][col]=num
 
            # return, if sucess, ya!
            if(solve_sudoku(arr)):
                return True
 
            # failure, unmake & try again
            arr[row][col] = 0
             
    # this triggers backtracking        
    return False

#--------------------------------------------------------------------------------------------------------

# Merge the related lines in the vector of tuples ('lines')
# Tuple present in 'lines' (rho, theta)
def mergeRelatedLines(lines, img):
    m,n = lines.shape
    size = img.shape
    for current in range(0, m):
        if(lines[current][0]==0) and (lines[current][1]==-100): 
            continue
        rho1 = lines[current][0]
        theta1 = lines[current][1]
        if(theta1>math.radians(45)) and (theta1<math.radians(135)):
            pt1current = (0, rho1/math.sin(theta1))
            pt2current = (size[1], -size[1]/math.tan(theta1) + rho1/math.sin(theta1)) 
        elif(theta1!=0):  
            pt1current = (rho1/math.cos(theta1), 0)
            pt2current = (-pt2current[1]/math.tan(theta1) + rho1/math.cos(theta1), size[0]) 
        for pos in range(0, n):
            if(lines[current][0]==lines[pos][0]) and (lines[current][1]==lines[pos][1]): 
                continue
            if(math.fabs(lines[pos][0]-lines[current][0])<20) and (math.fabs(lines[pos][1]-lines[current][1])<np.pi*10/180): 
                rho2 = lines[pos][0] 
                theta2 = lines[pos][1]

                if(lines[pos][1]>np.pi*45/180) and (lines[pos][1]<np.pi*135/180): 
                    pt1 = (0, rho2/math.sin(theta2)) 
                    pt2 = (size[1], -size[1]/math.tan(theta2) + rho2/math.sin(theta2)) 
                elif(theta2!=0): 
                    pt1 = (rho2/math.cos(theta2), 0)
                    pt2 = (-size[0]/math.tan(theta2) + rho2/math.cos(theta2), size[0])

                if((pt1[0]-pt1current[0])*(pt1[0]-pt1current[0]) + (pt1[1]-pt1current[1])*(pt1[1]-pt1current[1])<64*64) and ((pt2[0]-pt2current[0])*(pt2[0]-pt2current[0]) + ((pt2[1]-pt2current[1])*(pt2[1]-pt2current[1])<64*64)):
                    lines[current] = ((lines[current][0]+lines[pos][0])/2, (lines[current][1]+lines[pos][1])/2)
                    lines[pos]= (0, -100)
    return lines

# Draw the line on img
def drawLine(line,img,rgb):
    if line[1] != 0:
        m = int(-1/math.tan(line[1]))
        c = int(line[0]/math.sin(line[1]))
        cv2.line(img,(0,c),(width,m*width+c),rgb, 8)
    else:
        cv2.line(img,(line[0],0),(line[0],height),rgb,8)

originalImg = cv2.imread(str(sys.argv[1]), cv2.IMREAD_GRAYSCALE)  #read image grayscale
outerBox =  np.empty(originalImg.shape)
height, width = originalImg.shape 

processedImg = cv2.GaussianBlur(originalImg,(11,11),0)

processedImg = cv2.adaptiveThreshold(processedImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,5,2)
processedImg = cv2.bitwise_not(processedImg)

kernel = np.matrix([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
processedImg = cv2.dilate(processedImg, kernel, iterations=1)

count = 0
maxA = -1
maxPt = (0,0)

for y in range(width):
    for x in range(height):
        if processedImg[x][y] >= 128:
            area =  cv2.floodFill(processedImg,None,(y,x), cv2.cv.RGB(0,0,64))
            if area > maxA :
                maxA = area
                maxPt = (y,x)


retval, rect = cv2.floodFill(processedImg,None,maxPt, cv2.cv.RGB(255,255,255))

for y in range(width):
    for x in range(height):
        if (processedImg[x][y] == 64) and (x!=maxPt[1]) and (y!=maxPt[0]):
            area =  cv2.floodFill(processedImg,None,(y,x), cv2.cv.RGB(0,0,0))

processedImg = cv2.erode(processedImg, kernel, iterations=1)

lines = cv2.HoughLines(processedImg,1,np.pi/180,200)


lines[0] = mergeRelatedLines(lines[0], processedImg) # Merge the related lines found in the image in order to find the interception points of the edge of the board

for i in range(len(lines[0])):
    drawLine(lines[0][i],originalImg,cv2.cv.RGB(255,255,255)) # Draw the merged lines in the original image in order to make the grid merge with the bacground of the image

# Analyze the vector of lines to find the interceptions that represent the edges of the sudoku board
topEdge = (1000,1000)
topIntercept= (0, 100000)     

bottomEdge = (-1000,-1000)        
bottomIntercept= (0, 0)     

leftEdge = (1000,1000)    
leftIntercept = (100000, 0)     

rightEdge = (-1000,-1000)        
rightIntercept = (0, 0)

for i in range(len(lines[0])):               
    rho = lines[0][i][0]

    theta = lines[0][i][1]

    if(rho == 0) and (theta == -100):
        continue
    if(theta!=0):
        intercept = (rho/math.cos(theta), rho/(math.cos(theta)*math.sin(theta)))
        if(theta>math.radians(80)) and (theta<math.radians(100)):
            if(rho<topEdge[0]):
                topEdge = lines[0][i]
            if(rho>bottomEdge[0]):
                bottomEdge = lines[0][i]
        elif(theta<math.radians(10)) or (theta>math.radians(170)):
            if(intercept[0]>rightIntercept[0]):
                rightEdge = lines[0][i]
                rightIntercept = (intercept[0], rightIntercept[1])
            elif(intercept[0]<=leftIntercept[0]):
                leftEdge = lines[0][i]

                leftIntercept = (intercept[0], leftIntercept[1])



drawLine(topEdge, originalImg, cv2.cv.RGB(0,0,255))
drawLine(bottomEdge, originalImg, cv2.cv.RGB(0,0,255))
drawLine(leftEdge, originalImg, cv2.cv.RGB(0,0,255))
drawLine(rightEdge, originalImg, cv2.cv.RGB(0,0,255))


size = processedImg.shape
if(leftEdge[1]!=0):
    left1= (0, leftEdge[0]/math.sin(leftEdge[1]))
    left2= (size[1], -size[1]/math.tan(leftEdge[1]) + left1[1])
else:
    left1 = (leftEdge[0]/math.cos(leftEdge[1]), 0)
    left2 = (left1[0] - size[0]*math.tan(leftEdge[1]), size[0])

if(rightEdge[1]!=0):
    right1 = (0, rightEdge[0]/math.sin(rightEdge[1]))
    right2 = (size[1], -size[1]/math.tan(rightEdge[1]) + right1[1])
else:
    right1 = (rightEdge[0]/math.cos(rightEdge[1]), 0)
    right2 = (right1[0] - size[0]*math.tan(rightEdge[1]), size[0])

bottom1 = (0, bottomEdge[0]/math.sin(bottomEdge[1]))
bottom2 = (size[1], -size[1]/math.tan(bottomEdge[1]) + bottom1[1])     
top1 = (0, topEdge[0]/math.sin(topEdge[1]))     
top2 = (size[1], -size[1]/math.tan(topEdge[1]) + top1[1])

leftA = left2[1]-left1[1]
leftB = left1[0]-left2[0]     
leftC = leftA*left1[0] + leftB*left1[1]     
rightA = right2[1]-right1[1]     
rightB = right1[0]-right2[0]
rightC = rightA*right1[0] + rightB*right1[1]    
topA = top2[1]-top1[1]     
topB = top1[0]-top2[0]
topC = topA*top1[0] + topB*top1[1]     
bottomA = bottom2[1]-bottom1[1]     
bottomB = bottom1[0]-bottom2[0]
bottomC = bottomA*bottom1[0] + bottomB*bottom1[1]     

#Intersection of left and top
detTopLeft = leftA*topB - leftB*topA    
ptTopLeft = ((topB*leftC - leftB*topC)/detTopLeft, (leftA*topC - topA*leftC)/detTopLeft)     

# Intersection of top and right     
detTopRight = rightA*topB - rightB*topA   
ptTopRight = ((topB*rightC-rightB*topC)/detTopRight, (rightA*topC-topA*rightC)/detTopRight)

#Intersection of right and bottom     
detBottomRight = rightA*bottomB - rightB*bottomA
ptBottomRight = ((bottomB*rightC-rightB*bottomC)/detBottomRight, (rightA*bottomC-bottomA*rightC)/detBottomRight)

#Intersection of bottom and left
detBottomLeft = leftA*bottomB-leftB*bottomA
ptBottomLeft = ((bottomB*leftC-leftB*bottomC)/detBottomLeft, (leftA*bottomC-bottomA*leftC)/detBottomLeft)


maxLength = (ptBottomLeft[0]-ptBottomRight[0])*(ptBottomLeft[0]-ptBottomRight[0]) + (ptBottomLeft[1]-ptBottomRight[1])*(ptBottomLeft[1]-ptBottomRight[1])     
temp = (ptTopRight[0]-ptBottomRight[0])*(ptTopRight[0]-ptBottomRight[0]) + (ptTopRight[1]-ptBottomRight[1])*(ptTopRight[1]-ptBottomRight[1])
if(temp>maxLength):
    maxLength = temp
temp = (ptTopRight[0]-ptTopLeft[0])*(ptTopRight[0]-ptTopLeft[0]) + (ptTopRight[1]-ptTopLeft[1])*(ptTopRight[1]-ptTopLeft[1])
if(temp>maxLength):
    maxLength = temp
temp = (ptBottomLeft[0]-ptTopLeft[0])*(ptBottomLeft[0]-ptTopLeft[0]) + (ptBottomLeft[1]-ptTopLeft[1])*(ptBottomLeft[1]-ptTopLeft[1])
if(temp>maxLength):
    maxLength = temp
maxLength = math.sqrt(maxLength)


src = np.zeros((4, 2), dtype = "float32")
src[0][0] = ptTopLeft[0]
src[0][1] = ptTopLeft[1]
src[1][0] = ptTopRight[0]
src[1][1] = ptTopRight[1]
src[2][0] = ptBottomRight[0]
src[2][1] = ptBottomRight[1]
src[3][0] = ptBottomLeft[0]
src[3][1] = ptBottomLeft[1]

dst = np.zeros((4, 2), dtype = "float32")
dst[0][0] = 0
dst[0][1] = 0
dst[1][0] = maxLength -1
dst[1][1] = 0
dst[2][0] = maxLength -1
dst[2][1] = maxLength -1
dst[3][0] = 0
dst[3][1] = maxLength -1

# After the limits of the board are defined, the perspetive of the image is adjusted

M = cv2.getPerspectiveTransform(src, dst)
undistorted = cv2.warpPerspective(originalImg, M, (int(maxLength), int(maxLength)))

boxSize = int(maxLength/9)

# Knowing that the image contains only the sudoku board, we can assume that it is a 9x9 matrix and it can be cropped as so
# Each box is cropped and stored row-column.jpg

size = undistorted.shape
sudokuBoxes = np.zeros(shape=(9,9))
boxCount1 = 0
boxCount2 = 0
for i in range(0, size[0], boxSize):
    boxCount2 = 0
    for j in range(0,size[0], boxSize):
        if(i+boxSize<=size[0]) and (j+boxSize<=size[0]):
            img = undistorted[i:i+boxSize, j:j+boxSize]
            kernel = np.matrix([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            ratio2 = 3
            kernel_size = 3
            lowThreshold = 30
            img = cv2.blur(img, (1,1))
            strii = 'Boxes/'+str(boxCount1)+'-'+str(boxCount2) + '.jpg'
            cv2.imwrite(strii, img)
            boxCount2+= 1
    boxCount1+= 1

boxCount1= boxCount1-1


# In order to use the image to string function of the pytesseract, the image must be opened using the Image library and not the cv2
# The images of all the box are analyzed and the numbers are stored in the finalBoard matrix

finalBoard = np.zeros((9, 9))

boxCount1=0
for i in range(0, 9):
    boxCount2 = 0
    for j in range(0,9):
        strii = 'Boxes/'+str(boxCount1)+'-'+str(boxCount2) + '.jpg'
        strt = tes.image_to_string(Image.open(strii), config='-psm 10')
        if (strt.isdigit()):
            finalBoard[i][j] = int(strt)
        boxCount2+= 1
    boxCount1+= 1

# If the sudoku board has a solution it is printed on the terminal, otherwise it shows that there is no solution

if(solve_sudoku(finalBoard)):
    print_board(finalBoard)
else:
    print "No solution exists"

# Automated-garment-Measuring-System

import cv2
import numpy as np
 
cap = cv2.imread("C:/Users/Chiranthana/Pictures/m1.jpg")
 

frame=cap
hsv=cv2.cvtColor(cap,cv2.COLOR_BGR2HSV)

#blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
#hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

lower_blue = np.array([0, 0, 95])
upper_blue = np.array([250, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

contours, harr = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

for contour in contours:
    area = cv2.contourArea(contour)
   # print (area)
    if area > 5000:
        #print (area)
        cv2.drawContours(frame, contours, -1, (255, 255, 0), 1)
#print (frame)
new=frame
lenth = []
#frame[150][0][1]=2
#print (len(frame))
#print (frame[150][0])
cv2.imshow("Frame", frame)
cv2.imshow("Mask", mask)
for x in range (0,len(frame)):
    for y in range (0,len(frame[0])):
        if frame[x][y][0]==255 and frame[x][y][1]==255 and frame[x][y][2]==0:
            if 25<x<200 and 80<y<170 :
                new[x][y][0]=0
                new[x][y][1]=0
                new[x][y][2]=0
            else:
                new[x][y][0]=255
                new[x][y][1]=255
                new[x][y][2]=0
        else :
            new[x][y][0]=0
            new[x][y][1]=0
            new[x][y][2]=0


#print (frame[150])
cv2.imshow("new", new)
for x in range (0,len(frame)):
    for y in range (0,len(frame[0])):
        if new[x][y][0]==255 and new[x][y][1]==255 and new[x][y][2]==0:
            if x==150:
                 lenth.append(y)
print (new[150])
mesure=new
lenthx=str(2*(abs(lenth[0]-lenth[1])))+'mm'
cv2.line(mesure,(lenth[0],150),(lenth[1],150),(255,255,255),1)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(mesure,lenthx,(lenth[0]*2,150),font,0.5,(255,255,255),1,cv2.LINE_AA)
print (2*(abs(lenth[0]-lenth[1])))
cv2.imshow("mesure", mesure)
cap.release()
cv2.destroyAllWindows()

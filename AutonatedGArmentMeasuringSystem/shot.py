import cv2
import numpy as np	
import time
import math
cap = cv2.VideoCapture(0)
#set the width and height, and UNSUCCESSFULLY set the exposure time
cap.set(3,1280)
cap.set(4,1024)
cap.set(15, 0.1)
Pilen=0.10465
Fourcm=40


def DrawBoundry(bit_not):

    areaMax=0
    contours, harr = cv2.findContours(bit_not, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    Crrectboundry=[]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > areaMax:
            areaMax=area
            Crrectboundry=contour

    cv2.drawContours(frame, Crrectboundry, -1, (255, 255, 0), 1)
    for x in range (0,len(frame)):
        for y in range (0,len(frame[0])):
            if frame[x][y][0]==255 and frame[x][y][1]==255 and frame[x][y][2]==0:
                frame[x][y][0]==255
                frame[x][y][1]==255
                frame[x][y][2]==0
            else :
                frame[x][y][0]=0
                frame[x][y][1]=0
                frame[x][y][2]=0
    return frame

def templateMaching(img_detect):
    centerRect=[]
    img_gray = cv2.cvtColor(img_detect, cv2.COLOR_BGR2GRAY) 
    template = cv2.imread('C:/Users/Chiranthana/Pictures/c.jpg',0) 
    w, h = template.shape[::-1] 
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED) 
    threshold = 0.75######
    loc = np.where( res >= threshold)   
    for pt in zip(*loc[::-1]):
        if 300<pt[0]<700:
            cv2.rectangle(img_detect, pt, (pt[0] + w, pt[1] + h), (0,255,255), 1)
            centerRect=[pt[1]+20,pt[0]+20]
            
            break
    cv2.imshow('Detected',img_detect)
    if len(centerRect)!= 2:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,"Error on the clothes ajust it",(10,10),font,0.5,(255,255,255),1,cv2.LINE_AA)
        return None
    print("fsfvsssssss",centerRect)
    return centerRect
def drawlineAndvalue(datalist,frame,string):
    global y
    if len(datalist)!= 4:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,"Error on the clothes ajust it",(10,10),font,0.5,(255,255,255),1,cv2.LINE_AA)
        return None
    distance = math.sqrt( ((datalist[0]-datalist[2])**2)+((datalist[1]-datalist[3])**2) )
    lenthx=str(round(Pilen*(abs(distance)),2))+'cm'
    data = str(string+" "+"--"+" "+lenthx)
    y=int(y-15)
    cv2.line(frame,(datalist[1],datalist[0]),(datalist[3],datalist[2]),(255,255,255),1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,lenthx,(int((datalist[1]+datalist[3])/2),int((datalist[0]+datalist[2])/2)),font,0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(frame,data,(10,y),font,0.5,(255,0,255),1,cv2.LINE_AA)



            
def getlenthFrontLength(centerRect,frame):
    datalist=[]
    cv2.circle(frame,(centerRect[1],centerRect[0]), 5, (0,0,255), -1)
    for x in range(0,centerRect[0]):
        if frame[x][centerRect[1]][0]==255 and frame[x][centerRect[1]][1]==255 and frame[x][centerRect[1]][2]==0:
            datalist=[centerRect[0],centerRect[1],x,centerRect[1]]
    drawlineAndvalue(datalist,frame,"FrontLength")
    if len(datalist)!= 4:
        return None
    return datalist
  
def getlenthSleeveright(centerRect,frame):
    datalist=[]
    brk=0
    for y in range(0,len(frame[0])):
        for x in range(0,len(frame)):
            if frame[x][y][0]==255 and frame[x][y][1]==255 and frame[x][y][2]==0:
                datalist.append(x)
                datalist.append(y)
                brk=1
                break
        if brk==1:
            break
    for x in range(1,len(frame)):
        for y in range(0,centerRect[1]):
            if frame[len(frame)-x][y][0]==255 and frame[len(frame)-x][y][1]==255 and frame[len(frame)-x][y][2]==0:
                datalist.append(len(frame)-x)
                datalist.append(y)
                brk=0
                break
        if brk==0:
            break
    print("right",datalist)
    drawlineAndvalue(datalist,frame,"lenthSleeveright")
    return datalist
def getlenthSleeveleft(centerRect,frame):
    datalist=[]
    brk=0
    for y in range(1,len(frame[0])):
        for x in range(1,len(frame)):
            if frame[x][len(frame[0])-y][0]==255 and frame[x][len(frame[0])-y][1]==255 and frame[x][len(frame[0])-y][2]==0:
                datalist.append(x)
                datalist.append(len(frame[0])-y)               
                brk=1
                break
        if brk==1:
            break
    for x in range(1,len(frame)):
        for y in range(1,len(frame[0])-centerRect[1]):
            if frame[len(frame)-x][len(frame[0])-y][0]==255 and frame[len(frame)-x][len(frame[0])-y][1]==255 and frame[len(frame)-x][len(frame[0])-y][2]==0:
                datalist.append(len(frame)-x)
                datalist.append(len(frame[0])-y)
                brk=0
                break
        if brk==0:
            break
    print("left",datalist,len(frame))
    drawlineAndvalue(datalist,frame,"lenthSleeveleft")
    return datalist  
    
def getlenthbuttomleft(upValue,rightdata,frame):
    print(upValue)
    datalist=[int(upValue[2]),int(upValue[3]),int(rightdata[0]),int(rightdata[1])]
    drawlineAndvalue(datalist,frame,"lenthHem")
    
def getlenthbuttomright(upValue,leftdata,frame):
    print(upValue)
    datalist=[int(upValue[0]),int(upValue[1]),int(leftdata[0]),int(leftdata[1])]


    drawlineAndvalue(datalist,frame,"lenthChest")
##    
##def getnecktoUp(centerRect,frame):
##    datalist=[]
##    for x in range(0,centerRect[0]):
##        if frame[x][centerRect[1]][0]==255 and frame[x][centerRect[1]][1]==255 and frame[x][centerRect[1]][2]==0:
##            datalist=[centerRect[0],centerRect[1]+3,x,centerRect[1]+3]
##    drawlineAndvalue(datalist,frame,"necktoUp")

def waistlenth(centerRect,frame):
    datalist=[]
    brk=0
    cv2.circle(frame,(centerRect[1],centerRect[0]), 5, (0,0,255), -1)
    for x in range(0,centerRect[0]):
        for y in range(0,centerRect[1]):
            if frame[x][y][0]==255 and frame[x][y][1]==255 and frame[x][y][2]==0:
                datalist.append(x)
                datalist.append(y)
                brk=1
                break
        if brk==1:
            break
    for x in range(0,centerRect[0]):
        for y in range(centerRect[1],1024):
            if frame[x][y][0]==255 and frame[x][y][1]==255 and frame[x][y][2]==0:
                datalist.append(x)
                datalist.append(y)
                brk=0
                break
        if brk==0:
            break
    drawlineAndvalue(datalist,frame,"necklenth")
    return datalist
    
def erroroccured(measure):    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(measure,"put the mark corrly or ajust the colthes",(10,10),font,0.5,(255,255,255),3,cv2.LINE_AA)
    cv2.imshow("Error", measure)
    time.sleep(10)
    cv2.destroyAllWindows()
    
while True:
    y=700
    ret, frame = cap.read()
    ret,measure= cap.read()
    blurred_frame=cv2.GaussianBlur(frame,(3,3),0)
    hsv=cv2.cvtColor(blurred_frame,cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0, 0, 40])####
    upper_blue = np.array([250, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    bit_not = cv2.bitwise_not(mask)
    

    
    conerframe=DrawBoundry(bit_not)
    cv2.imshow("finalData", conerframe)
##    if (conerframe)==None:
##        erroroccured(measure)
##        continue
    print("trying")

    frame=DrawBoundry(bit_not)
    #cv2.imshow("boundry", frame)
    #Ajusted=clothAjusting(frame)
    #cv2.imshow("inpuAt", Ajusted)
    centerRect=templateMaching(measure)
    if (centerRect)==None:
        erroroccured(measure)
        continue
    waistpoint=waistlenth(centerRect,frame)
    
    if (waistpoint)==None:
        erroroccured(measure)
        continue
    rightdata=getlenthSleeveright(centerRect,frame)
    leftdata=getlenthSleeveleft(centerRect,frame)

    getlenthbuttomright(waistpoint,rightdata,frame)
    getlenthbuttomleft(waistpoint,leftdata,frame)
    

    cv2.imshow("finalData", frame)



    key = cv2.waitKey(10)
    if key == 27:
        break

cv2.destroyAllWindows() 

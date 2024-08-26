import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *


cap=cv2.VideoCapture('../videos/cars.mp4')

model=YOLO("../yolo_weights/yolov8n.pt")


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask=cv2.imread("mask.png")

#! define a tracker
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)

limits=[400,297,673,297]

total_count=[]

while True:
    success,img=cap.read()
    print("img",img)
    
    imgregion=cv2.bitwise_and(img , mask)
    results = model(imgregion, stream=True)
    
    detections=np.empty((0,5))
    
    
    
    print("results",results)
    for r in results:
        print("r is",r)
        boxes=r.boxes
        print("boxes",boxes)
        
        for box in boxes:
            print("box",box)
            #* open cv
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int (y1),int (x2),int(y2)
            # print("x1,y1,x2,y2",x1,y1,x2,y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(128, 0, 128),3)
            #& cvzone
            #^ x1,y1 are center of bounding boxes in this x1,y1,w,h=box.xywh[0] so calculation goes wrong when u pass to cvzone.cornerRect(img,bbox)
            # x1,y1,w,h=box.xywh[0]
            
            # bbox=int(x1),int (y1),int (w),int(h)
            
            #~ bbox is x1 ,y1 is top left corner 
            
            w,h=x2-x1,y2-y1
            # cvzone.cornerRect(img,(x1,y1,w,h),l=15,t=1)
            confidence=math.ceil((box.conf[0]*100))/100
            print("confidence",confidence)
            
            #* class name
            classes=int(box.cls[0])
            #^ there are many types of vehicles we want to detect only cars in that traffic
            currentclass=classNames[classes]
            
            if currentclass == "car" or currentclass =="bus" or currentclass =="truck" or currentclass =="motorbike"  and confidence >0.3:
                # cvzone.cornerRect(img,(x1,y1,w,h),l=15,t=1,rt=5)
                # cvzone.putTextRect(img,f'{classNames[classes]} {confidence}',(max(0,x1),max(35,y1)) ,scale=.6, thickness=1,offset=3)
                currentArray=np.array([x1,y1,x2,y2,confidence])
                detections=np.vstack((detections,currentArray))
            
    
    resulttracker=tracker.update(detections)
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,254),5)
    
    print("result tracker",resulttracker)
       
    for result in resulttracker:
        x1,y1,x2,y2,id=result
        id=int(id)
        x1,y1,x2,y2=int(x1),int (y1),int (x2),int(y2)
        print("result",result)
        w,h=x2-x1,y2-y1
        cvzone.cornerRect(img,(x1,y1,w,h),l=15,t=1,rt=5,colorR=(255,0,255))
        cvzone.putTextRect(img,f'{classNames[classes]} {id}',(max(0,x1),max(35,y1)) ,scale=2, thickness=3 ,offset=10)
        
        #& center point should touch the line
        cx,cy=x1+w//2,y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)
        
        if limits[0] <cx < limits[2] and   limits[1]-10 < cy < limits[3]+10:
            if total_count.count(id)==0:
                total_count.append(id)
                #* if it is detected make green line
                cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),5)
                
    
    
    # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.putText(img,str(len(total_count)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(255,165,0),8)
 
    
    
    
    
    print("results",results)
    print("image",img)
    cv2.imshow("Image",img)
    # cv2.imshow("Imageregion",imgregion)
    cv2.waitKey(0)




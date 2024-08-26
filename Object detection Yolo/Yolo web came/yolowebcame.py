from ultralytics import YOLO
import cv2
import cvzone
import math

cap=cv2.VideoCapture(0) 
cap.set(3,1280)
cap.set(4,720)
# cap=cv2.VideoCapture('../videos/people.mp4')

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




while True:
    success,img=cap.read()
    results = model(img, stream=True)
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
            cvzone.cornerRect(img,(x1,y1,w,h))
            confidence=math.ceil((box.conf[0]*100))/100
            print("confidence",confidence)
            
            #* class name
            classes=int(box.cls[0])
            
            
            cvzone.putTextRect(img,f'{classNames[classes]} {confidence}',(max(0,x1),max(35,y1)) ,scale=1, thickness=1)
            
      
            
    
    print("results",results)
    print("image",img)
    cv2.waitKey(1)
    cv2.imshow("image",img)



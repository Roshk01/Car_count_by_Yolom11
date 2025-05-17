import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
 
cap = cv2.VideoCapture("Video/4.mp4")  # For Video
 
model = YOLO("models/yolo11n.pt")

#for capture the random frame for measurement
# cap.set(cv2.CAP_PROP_POS_FRAMES, 20)
# success, frame = cap.read()
# if success:
#     cv2.imwrite("frame_for_measurement.png", frame)
#     print("Frame saved successfully.")
 
mask = cv2.imread("mask.png")
 
# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
 
limits = [1000,1446, 2810,1425]
totalCount = []
 
while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
 
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    results = model(imgRegion, stream=True)
 
    detections = np.empty((0, 5))
 
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
 
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            label = model.names[int(box.cls)]
 
            if label == "car" and conf > 0.3:
                # cvzone.putTextRect(img, f'{label} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
 
    resultsTracker = tracker.update(detections)
 
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (225, 40, 0), 5)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=3, colorR=(225, 0, 255))
        cvzone.putTextRect(img, f' {id}', (max(0, x1), max(35, y1)),scale=2, thickness=3)
 
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 7, (0, 225, 0),thickness=7)
 
        if limits[0] < cx < limits[2] and limits[1] - 30 < cy < limits[1] + 30:
            if totalCount.count(id) == 0:
                totalCount.append(id)
 
    cv2.putText(img,str(len(totalCount)),(425,150),cv2.FONT_HERSHEY_PLAIN,8,(50,50,255),15)
 
    img_resized = cv2.resize(img, None, fx=0.4, fy=0.4)  # resize the video
    cv2.imshow("Image", img_resized)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
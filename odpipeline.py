import cv2
import numpy as np
import cvzone
from cvzone.ColorModule import ColorFinder
thres = 0.45 # Threshold to detect object

hsvVals = {'hmin': 2, 'smin': 96, 'vmin': 86, 'hmax': 20, 'smax': 199, 'vmax': 255}

ballColorFinder = ColorFinder(False)

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

classNames= []
classFile = "coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,frame = cap.read()
    black = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

    imageColor, mask = ballColorFinder.update(frame, hsvVals)

    blurFrame = cv2.GaussianBlur(mask, (17, 17), 0)
    contours, _ = cv2.findContours(blurFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(black, contours, -1, (0,255,0), 3)

    classIds, confs, bbox = net.detect(black ,confThreshold=thres)
    # print(classIds,bbox)

    if len(classIds) != 0:
        for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(frame,box,color=(0,255,0),thickness=2)
            cv2.putText(frame,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(frame,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow("Output",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
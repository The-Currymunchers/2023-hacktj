import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder
import cv2 as cv

videoCapture = cv.VideoCapture(0)
prevCircle = None
def dist(x1, y1, x2, y2): return (x1-x2)**2+(y1-y2)**2

hsvVals = {'hmin': 2, 'smin': 96, 'vmin': 86, 'hmax': 20, 'smax': 199, 'vmax': 255}

ballColorFinder = ColorFinder(False)

while True:
    ret, frame = videoCapture.read()
    if not ret:
        break

    black = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

    imageColor, mask = ballColorFinder.update(frame, hsvVals)

    blurFrame = cv.GaussianBlur(mask, (17, 17), 0)
    contours, _ = cv.findContours(blurFrame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(black, contours, -1, (0,255,0), 3)

    if contours:
        c = max(contours, key=cv.contourArea)

        # Get the bounding box of the largest contour
        x, y, w, h = cv.boundingRect(c)
        
        # Draw a green circle around the volleyball
        cv.circle(frame, (x + w//2, y + h//2), w//2, (0,255,0), -1)

    imageStack = cvzone.stackImages([frame, blurFrame, black], 3, 0.6)

    cv.imshow("frame", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv.destroyAllWindows
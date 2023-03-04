import numpy as np
import cv2 as cv
import cvzone
from cvzone.ColorModule import ColorFinder

videoCapture = cv.VideoCapture(0)
prevCircle = None
def dist(x1, y1, x2, y2): return (x1-x2)**2+(y1-y2)**2

ballColorFinder = ColorFinder(True)


# This while loop sets up the camera to be used
while True:
    ret, frame = videoCapture.read()
    if not ret:
        break

    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurFrame = cv.GaussianBlur(grayFrame, (17, 17), 0)

    circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.2, 100,
                              param1=100, param2=30, minRadius=20, maxRadius=200)

    # The below statements are for selecting the best circle from the circles found in the Hough Circle Transform
    # The Hough Circle Transform is a basic feature extraction technique in image processing for detecting circles.
    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen = None
        for i in circles[0, :]:
            if chosen is None:
                chosen = i
            if prevCircle is not None:
                if dist(chosen[0], chosen[1], prevCircle[0], prevCircle[1]) <= dist(i[0], i[1], prevCircle[0], prevCircle[1]):
                    chosen = i
        cv.circle(frame, (chosen[0], chosen[1]), 1, (0, 100, 100), 3)
        cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (255, 0, 255), 3)
        prevCircle = chosen

    cv.imshow("circles", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv.destroyAllWindows

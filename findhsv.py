import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder
import cv2 as cv

hsvVals = {'hmin': 0, 'smin': 122, 'vmin': 174, 'hmax': 43, 'smax': 255, 'vmax': 255}

ballColorFinder = ColorFinder(False)

while True:
    image = cv.imread("../2023-HACKTJ/ball_image.png")
    imageColor, mask = ballColorFinder.update(image, hsvVals)

    imageStack = cvzone.stackImages([image, imageColor], 2, 0.5)

    cv.imshow("image", imageStack)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

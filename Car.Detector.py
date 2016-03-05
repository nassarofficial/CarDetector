import cv2
import numpy as np
import math

backsub = cv2.BackgroundSubtractorMOG2(300, 50, False)

avgofhul = []
avgofsol = []
avgofcirc = []

capture = cv2.VideoCapture("rheinhafen.mpg")
faceCascade = cv2.CascadeClassifier('cars3.xml')

def avger(arr):
    if len(arr) == 0:
        return 0
    else:
        return sum(arr) / len(arr)


def GetHullArea(h):
    lines = np.hstack([pts, np.roll(pts, -1, axis=0)])
    area = 0.5 * abs(sum(x1 * y2 - x2 * y1 for x1, y1, x2, y2 in lines))
    return area

def Haar(frame):
    roi = frame[0:350, 250:700]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame


def Contour(mask,frame):
    kernel = np.ones((13, 9), np.uint8)

    masked = cv2.bitwise_and(frame, frame, mask=mask)
    median = cv2.medianBlur(masked, 5)
    fgmask = backsub.apply(median, None, 0.01)

    fgmask = fgmask.copy()
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("TT", fgmask)

    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    try:
        hierarchy = hierarchy[0]

    except:
        hierarchy = []
    for contour, hier in zip(contours, hierarchy):

        (x, y, w, h) = cv2.boundingRect(contour)

        if w > 30 and h > 30:
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)

            solidity = float(area) / hull_area
            perimeter = cv2.arcLength(contour,True)
            circularity = (4*area*np.pi)/math.pow(perimeter, 2)

            avgofhul.append(hull_area)
            avgofsol.append(solidity)
            avgofcirc.append(circularity)


            print "test area ", area
            print " AVERAGE OF AREA " , avger(avgofhul)
            print " AVERAGE OF SOLIDITY " , avger(avgofsol)
            print " AVERAGE OF CIRCULARITY " , avger(avgofcirc)
            print "-----------------------------------"

            if circularity > 0.19 and solidity > 0.6 and hull_area > 3900:
                cv2.drawContours(frame, [hull], 0, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    ourarr = [frame,fgmask]
    return ourarr


if capture:
    while True:
        ret, frame = capture.read()
        if ret:
            #mask = cv2.imread('mask.png', 0)
            mask = cv2.imread('mask4.png', 0)

            outContour = Contour(mask,frame)


            ## Haar
            frame = Haar(outContour[0])

            cv2.imshow("Masked", outContour[1])
            cv2.imshow("Track", frame)

            key = cv2.waitKey(10)
            if key == ord('q'):
                break

import cv2
import time
import numpy as np 
import HandTrackingModule as htm
import math


detector = htm.handDetector(detectionCon=0.7)

cap = cv2.VideoCapture(0)
wCam, hCam = 780, 450
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

while True:
    success, img = cap.read()
    detector.findHands(img) 
    lmList = detector.findPosition(img, draw = False)
    if len(lmList) != 0:
        #print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img,(x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        print(length)

        #Changes the colour of the center point into a green colour 
        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)



    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
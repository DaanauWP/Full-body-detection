import cv2 
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

#previous time: 
pTime = 0
CTime = 0

while True:
    success, img = cap.read()

    # converting the image from BGR to RGB for hands
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # detecting all singular coordinates within the hand 
    if results.multi_hand_landmarks: 
        for handLms in results.multi_hand_landmarks:
            #printing landmarks
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                #height, width, channel 
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y *h)
                #print(id, cx, cy)
                #detecting the landmarks within the hand 
                if id == 4:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)


            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, 
                landmark_drawing_spec=mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=5),
                connection_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2))
     
    #calculating the fps
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


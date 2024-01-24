import cv2
import mediapipe as mp
import time
#import HandTrackingModule as htm

# previous time
pTime = 0
# Initialize the video capture from the default camera (index 0)
cap = cv2.VideoCapture(0)
    
    

detector = htm.handDetector()

while True:
    # Read a frame from the camera
    success, img = cap.read()
    if not success:
        print("Error: Could not read a frame from the camera.")
        break
    #apply draw function if needed

    img = detector.findHands(img)
    lmList = detector.findPosition(img)
        
    if lmList is not None and len(lmList) >= 5:
        #landmark number 4
        print(lmList[4])

    # calculating the frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # placing the frame rate within the screen 
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Display the image with landmarks
    cv2.imshow("Image", img)

    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
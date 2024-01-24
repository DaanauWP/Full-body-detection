import cv2
import mediapipe as mp
import time

class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Initialize the MediaPipe Hands module
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        # Initialize the MediaPipe Drawing Utilities
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # Convert the image from RGB to BGR (OpenCV uses BGR color format)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Process the image to detect hands
        self.result = self.hands.process(imgRGB)

        # Check if any hand landmarks are detected
        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    # Change the cqolor of hand connections to green (BGR color format: [0, 255, 0])
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.mpDraw.DrawingSpec(color=(255, 0 , 0), thickness=2, circle_radius=4),
                        connection_drawing_spec=self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2))

        return img

    def findPosition(self, img, handNum=0, draw=True):
        #landmark list 
        lmList = []
        if self.result.multi_hand_landmarks:
            myhand = self.result.multi_hand_landmarks[handNum]
            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return lmList

def main():
    # previous time
    pTime = 0
    # current time 
    cTime = 0
    # Initialize the video capture from the default camera (index 0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Camera not found or could not be opened.")
        return

    detector = handDetector()

    while True:
        # Read a frame from the camera
        success, img = cap.read()
        if not success:
            print("Error: Could not read a frame from the camera.")
            break
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

if __name__ == "__main__":
    main()
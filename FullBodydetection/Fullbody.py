import cv2
import time
from HandTrackingModule import handDetector
from posestimationRE import PoseDetector 
from FdModule import FaceDetector

# Initialize the hand detector
hand_detector = handDetector()

# Initialize the pose detector
pose_detector = PoseDetector()

# Initialize the pose detector
face_detector = FaceDetector()

cap = cv2.VideoCapture(0)
pTime = 0

while True:

    # Read a frame from the camera
    success, img = cap.read()
    if not success:
        print("Error: Could not read a frame from the camera.")
        break

    # Detect hands using the hand detector
    img = hand_detector.findHands(img)
    lmList = hand_detector.findPosition(img)

    # Detect pose using the pose detector
    pose_results = pose_detector.findPose(img)
    # Get specific landmark positions
    lmListPose = pose_detector.findPosition(img, draw=False)

    # Detect face using the face detector
    img, bboxs = face_detector.find_faces(img)


    #if lmList is not None and len(lmList) >= 5:
        # Landmark number 4
        #print("Hand Landmarks:", lmList[4])

   #if pose_results:
        #lmListPose = pose_detector.findPosition(img, draw = False)

    #if lmListPose:
            #print("pose Landmarks:", lmListPose[14])
            #cv2.circle(img, (lmListPose[14][1], lmListPose[14][2]), 15, (0, 255, 0), cv2.FILLED)

    #if bboxs:
        #for bbox in bboxs:
            #print("Face Bounding Box:", bbox)

    # Calculate the frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Place the frame rate within the screen
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Display the image with landmarks
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

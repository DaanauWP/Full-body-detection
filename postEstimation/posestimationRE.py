from multiprocessing import connection
from turtle import color
import cv2
import mediapipe as mp
import time
#import HandTrackingModule as htm

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions
        self.pose = self.mp_pose.pose.Pose()
        self.mp_drawing = self.mp_pose.drawing_utils

    # Check if pose landmarks are detected
    def findPose(self, frame, draw=True):
        # Converting the frames from BGR to RGB
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process pose on the RGB image
        pose_results = self.pose.process(frame_RGB)
        if pose_results.pose_landmarks:
            # Draw landmarks on the frame
            self.mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, self.mp_pose.pose.POSE_CONNECTIONS,
                                           landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), circle_radius=10),
                                           connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
        return pose_results  # Add this line to return pose_results

    def findPosition(self, frame, draw=True):
        lmList = []
        pose_results = self.findPose(frame, draw=False)  # Call findPose to get the pose_results
        if pose_results.pose_landmarks:
            for id, lm in enumerate(pose_results.pose_landmarks.landmark):
                h, w, c = frame.shape
                print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return lmList

# Rest of your code remains unchanged

def main():
    pTime = 0
    cTime = 0

    pose_detector = PoseDetector()

    # Open the webcam (camera index 0)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        # Reading the frames
        _, frame = cap.read()

        # calculating the frame rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Get pose results
        pose_results = pose_detector.findPose(frame)

        # Get specific landmark positions
        lmList = pose_detector.findPosition(frame, draw=False)

        if lmList:
            print(lmList[14])  # Assuming lmList[14] is the landmark you're interested in

            # Draw a circle around a specific landmark
            cv2.circle(frame, (lmList[14][1], lmList[14][2]), 15, (255, 0, 0), cv2.FILLED)

        # Display the frame
        cv2.imshow("Pose Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
import mediapipe as mp 
import time

class poseDetector():

    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose

        # Use boolean values for detectionCon and trackingCon
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, 
                                     bool(self.detectionCon), bool(self.trackingCon))

        # Initialize results as None
        self.results = None

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS,
                                        landmark_drawing_spec=self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                        connection_drawing_spec=self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2))

        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results and self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                print(id, lm)
                # gives the exact pixel value
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return lmList

def main():
    video_path = '/Users/wayanprice/Desktop/shadow boxing.mp4'
    cap = cv2.VideoCapture(video_path)
    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("End of video. Exiting...")
            break
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw= False)
        print(lmList[14])
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (255, 0, 0), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

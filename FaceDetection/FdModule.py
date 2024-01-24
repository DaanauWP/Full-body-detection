from unittest import result
import cv2
import mediapipe as mp
import time

class FaceDetector:
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        # Initialize mediapipe modules
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        # Create a FaceDetection object with a confidence threshold of 0.75
        self.faceDetection = self.mpFaceDetection.FaceDetection(0.75)

    def find_faces(self, img, draw=True):
        # Convert the frame to RGB for mediapipe processing
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the frame with the face detection model
        result = self.faceDetection.process(imgRGB)
        #bouding boxes   
        bboxs = []
        if result.detections:
            for id, detection in enumerate(result.detections):
                # Draw the bounding box around the face
                self.mpDraw.draw_detection(img, detection)

                # Draw the bounding box and display the detection score
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.face_draw(img, bbox)

                    cv2.putText(img, f'Score: {int(detection.score[0]*100)}%', 
                                (bbox[0], bbox[1] - 20), 
                                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

        return img, bboxs

    def face_draw(self, img, bbox, l = 30, thickness = 7, rt = 1):
        x, y, w, h = bbox
        x1, y1 = x +  w, y + h
        cv2.rectangle(img, bbox, (255, 0, 252), rt)
        # Top left x, y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), thickness)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), thickness)
        #Top right x1, y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), thickness)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), thickness)
        # Bottom left x, y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), thickness)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), thickness)
        #Bottom right x1, y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), thickness)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), thickness)


        return img


def main():
    # Set up the webcam capture
    cap = cv2.VideoCapture(0)
    pTime = 0

    # Create an instance of FaceDetector
    detector = FaceDetector()

    while True:
        # Read a frame from the webcam
        success, frame = cap.read()

        # Find faces in the frame
        result_frame, bboxs = detector.find_faces(frame)
        print(bboxs)

        # Calculate and display the frames per second
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(result_frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

        # Display the result
        cv2.imshow("Face Detection", result_frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
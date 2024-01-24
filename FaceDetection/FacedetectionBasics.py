# Import necessary libraries
from unittest import result
import cv2
import mediapipe as mp
import time

# Set up the webcam capture
cap = cv2.VideoCapture(0) 
pTime = 0

# Initialize mediapipe modules
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils

# Create a FaceDetection object with a confidence threshold of 0.75
faceDetection = mpFaceDetection.FaceDetection(0.75)

while True:
    # Read a frame from the webcam
    success, img = cap.read()

    # Convert the frame to RGB for mediapipe processing
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the frame with the face detection model
    result = faceDetection.process(imgRGB)
    #print(result)

    # Check if any face is detected
    if result.detections:
        for id, detection in enumerate(result.detections):
            # Draw the bounding box around the face
            mpDraw.draw_detection(img, detection)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            
            # Draw the bounding box and display the detection score
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'Score: {int(detection.score[0]*100)}%', 
                        (bbox[0], bbox[1] - 20), 
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

    # Calculate and display the frames per second
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

    # Display the image with the drawn annotations
    cv2.imshow("Image", img)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

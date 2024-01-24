from turtle import circle
import cv2
import mediapipe as mp
import time 

cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
drawSpec = mpDraw.DrawingSpec(thickness = 1, circle_radius = 3)


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = faceMesh.process(imgRGB)
    if result.multi_face_landmarks:
        for facesLms in result.multi_face_landmarks:
            mpDraw.draw_landmarks(img, facesLms, mpFaceMesh.FACEMESH_TESSELATION, 
                drawSpec, drawSpec)
            
            
            for id, lm in enumerate(facesLms.landmark):
                #print(lm)
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                print(id, x, y)

             

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
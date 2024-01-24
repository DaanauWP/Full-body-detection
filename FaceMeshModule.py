from math import factorial
import cv2
import mediapipe as mp
import time 

class FaceMeshDetector():
    def __init__(self, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.maxFaces = int(maxFaces)
        self.minDetectionCon = float(minDetectionCon)
        self.minTrackCon = float(minTrackCon)

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            max_num_faces=self.maxFaces, 
            min_detection_confidence=self.minDetectionCon, 
            min_tracking_confidence=self.minTrackCon
        )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=3)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.faceMesh.process(imgRGB)
        #multiple faces 
        faces = []
        if result.multi_face_landmarks:
            for facesLms in result.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, facesLms, self.mpFaceMesh.FACEMESH_TESSELATION, 
                        self.drawSpec, self.drawSpec)

                face = []
                for id, lm in enumerate(facesLms.landmark):
                    #print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    #applys each number value to each point
                    cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    #print(id, x, y)
                    face.append([x, y])
                faces.append(face)
        return img, faces

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()
    
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if len(faces) != 0:

            print(faces[0])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

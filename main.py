import cv2
import time
import numpy as np
from detector import FaceMeshDetector

# Constants
WIDTH, HEIGHT = 1280, 720
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.5
# Indices for eyes and mouth
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 78, 308] # Top, Bottom, Left, Right

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)
    
    detector = FaceMeshDetector(max_num_faces=1)
    
    pTime = 0
    
    while True:
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        img, faces = detector.findFaceMesh(img)
        
        if faces:
            face = faces[0]
            
            # Calculate EAR
            left_ear = detector.calculate_EAR(face, LEFT_EYE)
            right_ear = detector.calculate_EAR(face, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Calculate MAR
            mar = detector.calculate_MAR(face, MOUTH)
            
            # Determine State
            if avg_ear < EAR_THRESHOLD:
                status = "SLEEPING"
                color = (0, 0, 255) # Red
            elif mar > MAR_THRESHOLD:
                status = "YAWNING"
                color = (0, 255, 255) # Yellow
            else:
                status = "AWAKE"
                color = (0, 255, 0) # Green
                
            # Display Status and Values
            cv2.putText(img, f'Status: {status}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            cv2.putText(img, f'EAR: {avg_ear:.2f}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, f'MAR: {mar:.2f}', (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (WIDTH - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Drowsiness Detection", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

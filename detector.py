import cv2
import mediapipe as mp
import math

class FaceMeshDetector:
    def __init__(self, static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=self.static_image_mode,
            max_num_faces=self.max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        self.img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(self.img_rgb)
        faces = []
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, 
                        face_landmarks, 
                        self.mp_face_mesh.FACEMESH_TESSELATION,
                        self.draw_spec, 
                        self.draw_spec
                    )
                
                face = []
                ih, iw, ic = img.shape
                for id, lm in enumerate(face_landmarks.landmark):
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
        return img, faces

    def euclidean_distance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def calculate_EAR(self, face, eye_indices):
        # eye_indices: [p1, p2, p3, p4, p5, p6]
        # Vertical distances
        A = self.euclidean_distance(face[eye_indices[1]], face[eye_indices[5]])
        B = self.euclidean_distance(face[eye_indices[2]], face[eye_indices[4]])
        # Horizontal distance
        C = self.euclidean_distance(face[eye_indices[0]], face[eye_indices[3]])
        
        ear = (A + B) / (2.0 * C)
        return ear

    def calculate_MAR(self, face, mouth_indices):
        # mouth_indices: [top, bottom, left, right]
        # Vertical distance
        A = self.euclidean_distance(face[mouth_indices[0]], face[mouth_indices[1]])
        # Horizontal distance
        B = self.euclidean_distance(face[mouth_indices[2]], face[mouth_indices[3]])
        
        mar = A / B
        return mar

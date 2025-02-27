import face_recognition
import cv2
import numpy as np
import sqlite3

class FaceRecognizer:
    def _init_(self):
        self.known_face_encodings = []
        self.known_face_ids = []
        self.load_known_faces()

    def load_known_faces(self):
        """Load all registered faces from database"""
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("SELECT id, image_path FROM students")
        students = c.fetchall()
        
        self.known_face_encodings = []
        self.known_face_ids = []
        
        for student_id, image_path in students:
            try:
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)[0]
                self.known_face_encodings.append(encoding)
                self.known_face_ids.append(student_id)
            except Exception as e:
                print(f"Error loading image for student {student_id}: {str(e)}")
        
        conn.close()

    def recognize_face(self, frame):
        """Recognize faces in a video frame"""
        try:
            # Convert frame from BGR to RGB
            rgb_frame = frame[:, :, ::-1]
            
            # Find all face locations and encodings
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding in face_encodings:
                # Compare faces with known encodings
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                student_id = None

                # Find the best match
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    student_id = self.known_face_ids[best_match_index]

                return student_id
        except Exception as e:
            print(f"Face recognition error: {str(e)}")
            return None

        return None
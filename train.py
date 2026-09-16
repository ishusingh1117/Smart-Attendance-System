
import cv2
import os
import numpy as np

faces = []
labels = []

for student_id in os.listdir("students"):
    folder = f"students/{student_id}"

    for file in os.listdir(folder):
        image = cv2.imread(f"{folder}/{file}",
                           cv2.IMREAD_GRAYSCALE)

        if image is not None:
            faces.append(image)
            labels.append(int(student_id))

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

recognizer.save("model.yml")

print("Model trained successfully!")
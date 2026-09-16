
import cv2
import csv
import os
from datetime import datetime

# Load face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model.yml")

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Load student names
students = {}

with open("students.csv", newline="") as f:
    for row in csv.DictReader(f):
        students[int(row["ID"])] = row["Name"]

os.makedirs("attendance", exist_ok=True)
file = "attendance/records.csv"

if not os.path.exists(file):
    with open(file, "w", newline="") as f:
        csv.writer(f).writerow(
            ["ID", "Name", "Date", "Time"]
        )

marked = set()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        sid, confidence = recognizer.predict(face)
        print(sid, confidence)
        if confidence < 70 and sid in students:
            name = students[sid]
            text = name
            today = datetime.now().strftime("%Y-%m-%d")

            if (sid, today) not in marked:
                now = datetime.now()

                with open(file, "a", newline="") as f:
                    csv.writer(f).writerow([
                        sid, name, today,
                        now.strftime("%H:%M:%S")
                    ])

                marked.add((sid, today))
                print(f"Attendance marked: {name}")

        else:
            text = "Unknown"

        cv2.rectangle(frame, (x, y),
                      (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    cv2.imshow("Smart Attendance", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
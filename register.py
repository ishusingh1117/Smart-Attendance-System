
import cv2
import os

name = input("Enter student name: ")
student_id = input("Enter student ID: ")

folder = f"students/{student_id}"
os.makedirs(folder, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while count < 20:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        cv2.imwrite(f"{folder}/{count}.jpg", face)
        count += 1

        cv2.rectangle(frame, (x, y),
                      (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Register Student", frame)

    if cv2.waitKey(100) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Saved {count} face images for {name}")
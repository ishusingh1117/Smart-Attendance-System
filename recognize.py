







import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model.yml")

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        student_id, confidence = recognizer.predict(face)

        if confidence < 70:
            text = f"ID: {student_id}"
        else:
            text = "Unknown"

        cv2.rectangle(frame, (x, y),
                      (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

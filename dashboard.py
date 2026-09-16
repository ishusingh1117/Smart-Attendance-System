import cv2
import numpy as np
import streamlit as st
import pandas as pd
st.header("Register Student")

with st.form("register"):
    sid = st.text_input("Student ID")
    name = st.text_input("Student Name")

    submit = st.form_submit_button("Register")

    if submit:
        if sid and name:
            file = "students.csv"

            df = pd.read_csv(file)
            st.metric("Total Attendance", len(df))
            st.metric("Unique Students", df["ID"].nunique())
            new_student = pd.DataFrame(
                [{"ID": int(sid), "Name": name}]
            )

            df = pd.concat([df, new_student], ignore_index=True)
            df.to_csv(file, index=False)

            st.success("Student registered!")
        else:
            st.error("Enter both ID and name.")
            st.header("📸 Mobile Attendance")

photo = st.camera_input("Take your attendance photo")

if photo:
    st.image(photo, caption="Captured photo")
if photo:
    img = cv2.imdecode(
        np.frombuffer(photo.getvalue(), np.uint8),
        cv2.IMREAD_COLOR
    )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("model.yml")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    ).detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("No face detected. Try again.")

    for (x, y, w, h) in faces:
        student_id, confidence = recognizer.predict(
            gray[y:y+h, x:x+w]
        )
        st.write("Recognized ID:", student_id)
        st.write("Confidence:", confidence)

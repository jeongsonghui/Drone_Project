import cv2
import numpy as np

# 얼굴 탐지기 로드 (Haarcascade 사용)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# LBPH 얼굴 인식기 생성
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# 학습할 얼굴 데이터
faces = []
labels = []
label_id = 0  # 특정 인물에 대한 ID

# 웹캠 또는 동영상에서 얼굴 학습
cap = cv2.VideoCapture(0)  # 웹캠 사용

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in detected_faces:
        faces.append(gray[y:y+h, x:x+w])
        labels.append(label_id)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Training", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 얼굴 데이터 학습
face_recognizer.train(faces, np.array(labels))
face_recognizer.save("trained_model.yml")
print("얼굴 학습 완료")
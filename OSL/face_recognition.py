import cv2
import face_recognition

# 1. 기준이 될 이미지 로드 (이미 저장된 사람의 얼굴)
known_image = face_recognition.load_image_file("../Images/songhee.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]  # 얼굴 특징 추출

# 2. 새로운 이미지 로드 (실시간 웹캠)
cap = cv2.VideoCapture(0)  # 웹캠 실행

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. 현재 프레임에서 얼굴 찾기
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        # 4. 새로운 얼굴과 기존 얼굴 비교
        matches = face_recognition.compare_faces([known_encoding], face_encoding)
        name = "Unknown"

        if True in matches:
            name = "Matched Person"  # 등록된 사람일 경우

        # 5. 화면에 결과 표시
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

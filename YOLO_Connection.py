import cv2
import torch
import numpy as np
from ultralytics import YOLO
from djitellopy import Tello

# 🟢 YOLOv10 모델 로드 (best.pt)
model = YOLO("C:/best.pt")

# 🟢 Tello 드론 연결
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

# 🟢 비디오 스트리밍 시작
tello.streamon()
frame_read = tello.get_frame_read()

# 🟢 화면 크기 및 중앙 영역 설정
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CENTER_X = FRAME_WIDTH // 2
CENTER_Y = FRAME_HEIGHT // 2
TOLERANCE = 50  # 중앙에서 벗어난 허용 오차

# 🔵 [추가] 전처리 함수 (명암 대비 보정 & 크기 조정)
def preprocess_frame(frame):
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))  # 크기 맞추기
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 흑백 변환
    gray = cv2.equalizeHist(gray)  # 명암 대비 보정
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # 다시 컬러로 변환
    return frame

try:
    while True:
        # 1️⃣ 카메라에서 현재 프레임 읽기
        frame = frame_read.frame
        frame = preprocess_frame(frame)  # 🔵 전처리 적용

        # 2️⃣ YOLO 모델을 사용해 얼굴 감지
        results = model(frame)

        # 얼굴이 감지되었는지 확인
        face_detected = False
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 감지된 얼굴 좌표
                confidence = box.conf[0].item()  # 신뢰도
                
                if confidence > 0.5:  # 신뢰도 50% 이상만 표시
                    face_detected = True
                    x_center = (x1 + x2) // 2
                    y_center = (y1 + y2) // 2

                    # 감지된 얼굴에 박스 표시
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Face {confidence:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 3️⃣ 얼굴이 감지되지 않으면 로그 출력
        if not face_detected:
            print("얼굴 감지되지 않음")

        # 4️⃣ 화면 출력
        cv2.imshow("Tello Face Tracking", frame)
        
        # 'q'를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("수동 종료")

finally:
    # 🛑 드론 착륙 및 정리
    tello.streamoff()
    cv2.destroyAllWindows()

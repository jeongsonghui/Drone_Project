import cv2
import torch
import numpy as np
from ultralytics import YOLO
from djitellopy import Tello

# YOLOv10 모델 로드 (best.pt)
model = YOLO("../models/best.pt")  # best.pt 경로 확인

#Tello 드론 연결
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")  # 배터리 상태 출력

# 비디오 스트리밍 시작
tello.streamon()
frame_read = tello.get_frame_read()

# 화면 크기 설정
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# 클래스 리스트 (송희, 희지)
CLASS_NAMES = ["songhee", "heeji"]  # YOLO 학습 시 지정한 클래스 이름

# 색감 보정 함수
def adjust_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR → RGB 변환
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=20)  # 밝기 & 대비 조정
    return img

try:
    while True:
        # 카메라에서 현재 프레임 읽기
        frame = frame_read.frame
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        

        # 이미지 색감 보정 및 화면 좌우 반전
        frame = cv2.flip(frame, 1)
        frame = adjust_image(frame)
        

        #  YOLO 모델로 얼굴 감지 및 분류
        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
                confidence = box.conf[0].item()  # 신뢰도
                class_id = int(box.cls[0].item())  # 클래스 ID
                
                if confidence > 0.5:  # 신뢰도 50% 이상인 경우만 표시
                    label = CLASS_NAMES[class_id]  # 클래스 ID를 이름으로 변환
                    color = (0, 255, 0) if label == "songhee" else (255, 0, 0)  # 색상 구분

                    # 바운딩 박스 크기 조정 (너무 작거나 크면 인식 오류 발생 가능)
                    width = x2 - x1
                    height = y2 - y1
                    if width < 50 or height < 50:  
                        continue  # 너무 작은 객체 무시

                    # 바운딩 박스 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 화면 출력
        cv2.imshow("Tello Face Detection", frame)

        # 'q'를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("수동 종료")

finally:
    # 드론 종료 및 스트리밍 해제
    tello.streamoff()
    cv2.destroyAllWindows()

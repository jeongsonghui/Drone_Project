from djitellopy import Tello
import cv2
from ultralytics import YOLO

# 1. YOLOv10n 사전학습 모델 로컬에서 로드
model = YOLO('./models/yolov10n.pt')  # 인터넷 연결 없이 로컬 파일만 사용

# 2. 드론 초기화 및 연결
tello = Tello()
tello.connect()
print(f"배터리 상태: {tello.get_battery()}%")

tello.streamon()
frame_read = tello.get_frame_read()

# 3. 객체 탐지 루프
try:
    while True:
        frame = frame_read.frame
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 4. YOLO에 프레임 전달하여 추론
        results = model(frame)[0]  # 첫 번째 결과만 사용

        # 5. 탐지된 객체 중 'person'만 시각화
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = box.conf[0].item()
            label = model.names[cls_id]

            if label == 'person':  # 사람만 필터링
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("tello View - YOLOv10 People Detection", frame)

        # 종료 키: 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("종료합니다.")

# 6. 정리
tello.streamoff()
cv2.destroyAllWindows()

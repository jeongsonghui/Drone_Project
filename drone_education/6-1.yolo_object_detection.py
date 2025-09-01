from djitellopy import Tello
import cv2
from ultralytics import YOLO

model = YOLO('./models/yolov10n.pt')

tello = Tello()
tello.connect()
print(f"배터리 상태: {tello.get_battery()}%")

tello.streamon()
frame_read = tello.get_frame_read()

# [추가] 창 설정 (크기 조절 가능)
cv2.namedWindow("tello View - YOLOv10 People Detection", cv2.WINDOW_NORMAL)

try:
    while True:
        frame = frame_read.frame
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # ← 네가 쓰던 그대로 둠

        results = model(frame)[0]

        # [추가] 현재 프레임에서 탐지된 'person' 수 카운트용
        person_count = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = box.conf[0].item()
            label = model.names[cls_id]

            if label == 'person':
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # [추가] 좌상단에 person 개수 표시 (드론 화면 + 박스 보이게)
        cv2.putText(frame, f'person: {person_count}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("tello View - YOLOv10 People Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("종료합니다.")

tello.streamoff()
cv2.destroyAllWindows()
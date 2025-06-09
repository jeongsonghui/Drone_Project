from djitellopy import Tello
import cv2
from ultralytics import YOLO
import time

# 모델 로드
model = YOLO("./models/best4.pt")  # 리프터, 사람 분류

# 드론 초기화
tello = Tello()
tello.connect()
tello.streamon()

# 목표 클래스 이름
TARGET_CLASSES = ["person", "lifter"]  # 'person' 또는 'lifter'가 감지되면 출력
FRAME_WIDTH = 640

# 카메라 스트리밍만 활성화, 이륙하지 않음
while True:
    frame = tello.get_frame_read().frame
    frame = cv2.resize(frame, (FRAME_WIDTH, 480))
    
    results = model(frame, verbose=False)[0]
    
    target_found = False
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result
        class_id = int(cls)
        label = model.names[class_id]
        
        # 'person' 또는 'lifter'가 감지되었을 경우
        if label in TARGET_CLASSES:
            target_found = True
            print(f"{label} 감지됨! (confidence: {conf:.2f})")
            
            # 박스 그리기
            frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            frame = cv2.putText(frame, f"{label} ({conf:.2f})", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            break  # 하나만 추적

    if not target_found:
        print("리프터 또는 사람 탐지 안됨")
    
    # 화면 표시
    cv2.imshow("Tello Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
tello.streamoff()
cv2.destroyAllWindows()

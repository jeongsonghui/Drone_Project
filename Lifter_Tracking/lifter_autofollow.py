from djitellopy import Tello
import cv2
from ultralytics import YOLO
import threading
import queue
import time

model = YOLO("C:/Users/Administrator/Desktop/TelloDrone/models/best.pt")

tello = Tello()
tello.connect()
print("배터리:", tello.get_battery())
tello.streamon()

frame_read = tello.get_frame_read()

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CENTER_X = FRAME_WIDTH // 2
CENTER_Y = FRAME_HEIGHT // 2
TOLERANCE = 40

TARGET_LABEL = "lifter"
MIN_BOX_HEIGHT = 120
MAX_BOX_HEIGHT = 180

frame_queue = queue.LifoQueue(maxsize=10)

def fetch_frames():
    while True:
        frame = frame_read.frame
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(frame)
        time.sleep(0.05)

def process_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            results = model(frame, verbose=False)[0]
            yaw_velocity = 0
            up_down_velocity = 0
            forward_backward_velocity = 0
            target_found = False

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = result
                class_id = int(cls)
                label = model.names[class_id]

                if label == TARGET_LABEL:
                    target_found = True
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    box_height = int(y2 - y1)

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({conf:.2f})", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # 회전 제어 (좌우 중심 맞추기)
                    dx = cx - CENTER_X
                    if abs(dx) > TOLERANCE:
                        yaw_velocity = int(dx * 0.2)  # 중심 벗어난 정도에 비례해 회전
                        yaw_velocity = max(min(yaw_velocity, 40), -40)

                    # 상하 제어
                    dy = cy - CENTER_Y
                    if abs(dy) > TOLERANCE:
                        up_down_velocity = -int(dy * 0.2)  # 위쪽은 음수로 올라감
                        up_down_velocity = max(min(up_down_velocity, 40), -40)

                    # 거리 제어 (박스 크기 기준)
                    if box_height < MIN_BOX_HEIGHT:
                        forward_backward_velocity = 10
                    elif box_height > MAX_BOX_HEIGHT:
                        forward_backward_velocity = -10
                    else:
                        forward_backward_velocity = 0

                    break

            if not target_found:
                print("리프터 못 찾음")
                yaw_velocity = 0
                up_down_velocity = 0
                forward_backward_velocity = 0

            tello.send_rc_control(0, forward_backward_velocity, up_down_velocity, yaw_velocity)

            cv2.imshow("📷 Tello Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    tello.send_rc_control(0, 0, 0, 0)
    tello.land()
    tello.streamoff()
    cv2.destroyAllWindows()

# 이륙
tello.takeoff()
time.sleep(2)

fetch_thread = threading.Thread(target=fetch_frames, daemon=True)
process_thread = threading.Thread(target=process_frames)

fetch_thread.start()
process_thread.start()
process_thread.join()

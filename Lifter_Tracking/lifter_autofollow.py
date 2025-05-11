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

frame_queue = queue.LifoQueue(maxsize=20)

def fetch_frames():
    prev_time = time.time()
    frame_count = 0

    while True:
        frame = frame_read.frame
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(frame)
        frame_count += 1

        if frame_queue.qsize() > 20:
            print(f"[⚠️ Warning] frame_queue size 이상: {frame_queue.qsize()}")

        # IO FPS 로그
        if time.time() - prev_time >= 1.0:
            print(f"[📥 IO FPS] {frame_count} fps")
            frame_count = 0
            prev_time = time.time()

        time.sleep(0.05)

def process_frames():
    infer_count = 0
    infer_time = time.time()

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # 추론 시작
            results = model(frame, verbose=False)[0]
            infer_count += 1

            # 추론 FPS 로그
            if time.time() - infer_time >= 1.0:
                print(f"[🧠 Inference FPS] {infer_count} fps")
                infer_count = 0
                infer_time = time.time()

            yaw_velocity = 0
            up_down_velocity = 0
            forward_backward_velocity = 0
            target_found = False

            print(f"[▶️ 입력 프레임] size: {frame.shape}")
            print(f"[🧠 탐지 결과] 박스 {len(results.boxes)}개")

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = result
                class_id = int(cls)
                label = model.names[class_id]

                if label == TARGET_LABEL:
                    target_found = True
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    box_height = int(y2 - y1)

                    print(f"[🎯 타겟 탐지] {label} @ ({cx}, {cy}) / 높이: {box_height}")

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({conf:.2f})", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # 회전 제어
                    dx = cx - CENTER_X
                    if abs(dx) > TOLERANCE:
                        yaw_velocity = int(dx * 0.2)
                        yaw_velocity = max(min(yaw_velocity, 40), -40)

                    # 상하 제어
                    dy = cy - CENTER_Y
                    if abs(dy) > TOLERANCE:
                        up_down_velocity = -int(dy * 0.2)
                        up_down_velocity = max(min(up_down_velocity, 40), -40)

                    # 거리 제어
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

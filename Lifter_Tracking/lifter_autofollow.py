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

# ⏱️ IO 및 추론 프레임 속도 측정용
io_counter = 0
infer_counter = 0
last_io_time = time.time()
last_infer_time = time.time()

def fetch_frames():
    global io_counter, last_io_time
    while True:
        frame = frame_read.frame
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(frame)

        io_counter += 1
        current_time = time.time()
        if current_time - last_io_time >= 1.0:
            print(f"[📥 IO FPS] {io_counter} fps")
            io_counter = 0
            last_io_time = current_time

        time.sleep(0.01)

def process_frames():
    global infer_counter, last_infer_time
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            infer_start = time.time()
            results = model(frame, verbose=False)[0]
            infer_end = time.time()

            infer_counter += 1
            if infer_end - last_infer_time >= 1.0:
                print(f"[🧠 Inference FPS] {infer_counter} fps")
                infer_counter = 0
                last_infer_time = infer_end

            yaw_velocity = 0
            up_down_velocity = 0
            forward_backward_velocity = 0
            target_found = False

            print(f"[🧠 탐지결과] 박스 {len(results.boxes)}개")

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = result
                class_id = int(cls)
                label = model.names[class_id]
                confidence = float(conf)
                label_text = f"{label} ({confidence:.2f})"
                box_coords = f"({int(x1)}, {int(y1)}) ~ ({int(x2)}, {int(y2)})"

                # 콘솔 출력
                print(f"[🧠 탐지결과] {label} | 정확도: {confidence:.2f} | 바운딩박스: {box_coords}")

                if label == TARGET_LABEL:
                    target_found = True
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    box_height = int(y2 - y1)

                    # 프레임에 라벨과 박스 표시
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, label_text, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # 드론 이동 제어
                    dx = cx - CENTER_X
                    if abs(dx) > TOLERANCE:
                        yaw_velocity = int(dx * 0.2)
                        yaw_velocity = max(min(yaw_velocity, 40), -40)

                    dy = cy - CENTER_Y
                    if abs(dy) > TOLERANCE:
                        up_down_velocity = -int(dy * 0.2)
                        up_down_velocity = max(min(up_down_velocity, 40), -40)

                    if box_height < MIN_BOX_HEIGHT:
                        forward_backward_velocity = 10
                    elif box_height > MAX_BOX_HEIGHT:
                        forward_backward_velocity = -10
                    else:
                        forward_backward_velocity = 0

                    print(f"[🎯 타겟 탐지] {label} @ ({cx}, {cy}) / 높이: {box_height}")
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

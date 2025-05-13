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

# 프레임 수, 추론 수 카운터
io_counter = 0
infer_counter = 0
last_log_time = time.time()

# 탐지결과 저장용
latest_detections = []

def fetch_frames():
    global io_counter
    while True:
        frame = frame_read.frame
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(frame)
        io_counter += 1
        time.sleep(0.01)

def process_frames():
    global infer_counter, latest_detections, last_log_time, io_counter

    try:
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                results = model(frame, verbose=False)[0]
                infer_counter += 1

                yaw_velocity = 0
                up_down_velocity = 0
                forward_backward_velocity = 0
                target_found = False

                latest_detections = []  # 감지 결과 초기화

                for result in results.boxes.data.tolist():
                    x1, y1, x2, y2, conf, cls = result
                    class_id = int(cls)
                    label = model.names[class_id]
                    confidence = float(conf)
                    label_text = f"{label} ({confidence:.2f})"
                    box_coords = f"({int(x1)}, {int(y1)}) ~ ({int(x2)}, {int(y2)})"

                    latest_detections.append({
                        "label": label,
                        "conf": confidence,
                        "bbox": box_coords
                    })

                    if label == TARGET_LABEL:
                        target_found = True
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        box_height = int(y2 - y1)

                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, label_text, (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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

                        break  # 첫 번째 타겟만 추적

                if not target_found:
                    forward_backward_velocity = 0
                    up_down_velocity = 0
                    yaw_velocity = 0

                tello.send_rc_control(0, forward_backward_velocity, up_down_velocity, yaw_velocity)

                # 로그 출력: 1초마다
                current_time = time.time()
                if current_time - last_log_time >= 1.0:
                    print(f"\n[📥 IO FPS] {io_counter} fps | [🧠 Inference FPS] {infer_counter} fps")
                    print(f"[🧠 탐지결과] 총 {len(latest_detections)}개")
                    for det in latest_detections:
                        print(f"  └ {det['label']} | 정확도: {det['conf']:.2f} | 바운딩박스: {det['bbox']}")
                    if target_found:
                        print(f"[🎯 타겟 탐지] {TARGET_LABEL} 추적 중")
                    else:
                        print(f"[⚠️ 타겟 없음] {TARGET_LABEL} 감지 실패")

                    io_counter = 0
                    infer_counter = 0
                    last_log_time = current_time

                cv2.imshow("📷 Tello Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        # 비상 정지 및 종료 루틴
        tello.send_rc_control(0, 0, 0, 0)
        tello.land()
        tello.streamoff()
        cv2.destroyAllWindows()

# 이륙 및 쓰레드 시작
tello.takeoff()
time.sleep(2)

fetch_thread = threading.Thread(target=fetch_frames, daemon=True)
process_thread = threading.Thread(target=process_frames)

fetch_thread.start()
process_thread.start()
process_thread.join()

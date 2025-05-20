from djitellopy import Tello
import cv2
from ultralytics import YOLO
import threading
import queue
import time
import logging
from logging.handlers import QueueHandler, QueueListener

# Logging Setup 
log_queue = queue.Queue(-1)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("drone.log", mode='a')
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
queue_handler = QueueHandler(log_queue)
listener = QueueListener(log_queue, console_handler, file_handler)

logger = logging.getLogger("DroneLogger")
logger.setLevel(logging.INFO)
logger.addHandler(queue_handler)
listener.start()

# Model and Drone Setup
model = YOLO("C:/Users/Administrator/Desktop/Drone_Project/models/best4.pt")
tello = Tello()
tello.connect()
logger.info(f"배터리: {tello.get_battery()}%")
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

frame_queue = queue.Queue(maxsize=10)

# Thread FPS Counters
io_counter = 0
infer_counter = 0
last_io_time = time.time()
last_infer_time = time.time()

# Frame Fetch Thread
def fetch_frames():
    global io_counter, last_io_time
    while True:
        frame = frame_read.frame
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                frame_queue.get_nowait()
                frame_queue.put_nowait(frame)
            except queue.Empty:
                pass

        io_counter += 1
        if time.time() - last_io_time >= 1.0:
            logger.info(f"[IO FPS] {io_counter} fps")
            io_counter = 0
            last_io_time = time.time()

        time.sleep(0.01)

# Detection Thread
def process_frames():
    global infer_counter, last_infer_time
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            results = model(frame, verbose=False)[0]

            infer_counter += 1
            if time.time() - last_infer_time >= 1.0:
                logger.info(f"[Inference FPS] {infer_counter} fps")
                infer_counter = 0
                last_infer_time = time.time()

            yaw_velocity = 0
            up_down_velocity = 0
            forward_backward_velocity = 0
            target_found = False

            logger.info(f"[탐지결과] 박스 {len(results.boxes)}개")

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = result
                class_id = int(cls)
                label = model.names[class_id]
                confidence = float(conf)
                label_text = f"{label} ({confidence:.2f})"
                box_coords = f"({int(x1)}, {int(y1)}) ~ ({int(x2)}, {int(y2)})"

                logger.info(f"[탐지결과] {label} | 정확도: {confidence:.2f} | 바운딩박스: {box_coords}")

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

                    logger.info(f"[타겟 탐지] {label} @ ({cx}, {cy}) / 높이: {box_height}")
                    break

            if not target_found:
                logger.info("리프터 탐지 실패")
                yaw_velocity = 0
                up_down_velocity = 0
                forward_backward_velocity = 0

            try:
                tello.send_rc_control(0, forward_backward_velocity, up_down_velocity, yaw_velocity)
            except Exception as e:
                logger.warning(f"[명령 전송 오류] {e}")

            cv2.imshow("📷 Tello Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    tello.send_rc_control(0, 0, 0, 0)
    tello.land()
    tello.streamoff()
    cv2.destroyAllWindows()

# Main Start
try:
    tello.takeoff()
    time.sleep(2)

    fetch_thread = threading.Thread(target=fetch_frames, daemon=True)
    process_thread = threading.Thread(target=process_frames)

    fetch_thread.start()
    process_thread.start()
    process_thread.join()

except KeyboardInterrupt:
    logger.info("Ctrl+C에 의한 종료 요청 감지됨")
    tello.send_rc_control(0, 0, 0, 0)
    tello.land()
    tello.streamoff()
    cv2.destroyAllWindows()
    listener.stop()
    logger.info("모든 리소스 종료 완료")
    exit(0)

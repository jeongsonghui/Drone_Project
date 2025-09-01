import os
import cv2
import threading
import queue
import time
from ultralytics import YOLO
from djitellopy import Tello
from deep_sort_realtime.deepsort_tracker import DeepSort

# =============== 설정 ================
TARGET_CLASS   = 'person'
CONF_THRESHOLD = 0.4

FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480
CENTER_X       = FRAME_WIDTH  // 2
CENTER_Y       = FRAME_HEIGHT // 2
CENTER_MARGIN  = 50           # 데드존(픽셀)

# 거리(크기) 제어용 목표 박스 너비
DESIRED_WIDTH  = 150          # 상황에 맞게 조정

# P 게인 (필요 시 조정)
Kp_yaw = 0.30   # 회전 속도 게인
Kp_ud  = 0.40   # 상하 속도 게인
Kp_fb  = 0.45   # 전후 속도 게인

# 속도 제한 (-100 ~ 100)
MAX_YAW = 60
MAX_UD  = 60
MAX_FB  = 40

# 스무딩(EMA) 계수 (0~1, 클수록 부드럽고 반응은 느림)
SMOOTH_ALPHA = 0.6

WEIGHTS_PATH   = os.path.join(os.path.dirname(__file__), '..', 'yolov8n.pt')
QUEUE_MAXSIZE  = 10
# ====================================

# ============ YOLOv8 로드 ============
model = YOLO(WEIGHTS_PATH)
names = model.names
target_class_id = [k for k, v in names.items() if v == TARGET_CLASS][0]

# ============ Deep SORT ============
tracker = DeepSort(max_age=30)

# ============ DJI Tello ============
tello = Tello()
tello.connect()
print("배터리:", tello.get_battery(), "%")
tello.streamon()

# -------- 프레임 수집 스레드 + 큐 --------
frame_queue = queue.LifoQueue(maxsize=QUEUE_MAXSIZE)
stop_event = threading.Event()

io_counter = 0
infer_counter = 0
last_io_time = time.time()
last_infer_time = time.time()

def frame_reader_loop():
    global io_counter, last_io_time
    fr = tello.get_frame_read()
    while not stop_event.is_set():
        frame = fr.frame
        if frame is None:
            time.sleep(0.005)
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        if frame_queue.full():
            try:
                frame_queue.get_nowait()  # 오래된 프레임 드랍
            except queue.Empty:
                pass
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass

        io_counter += 1
        now = time.time()
        if now - last_io_time >= 1.0:
            print(f"[📥 IO FPS] {io_counter} fps")
            io_counter = 0
            last_io_time = now

        time.sleep(0.005)

reader_thread = threading.Thread(target=frame_reader_loop, daemon=True)
reader_thread.start()

# ======== 연속 RC 제어 준비 ========
# 초기 속도(이전값) — 스무딩용
prev_lr = 0
prev_fb = 0
prev_ud = 0
prev_yaw = 0

# 감쇠(타깃 없을 때 점진적 0으로 수렴)
DECAY = 0.85

# 이륙 & RC 초기화
tello.takeoff()
tello.send_rc_control(0, 0, 0, 0)

try:
    while True:
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            # 프레임 없으면 정지 유지
            tello.send_rc_control(0, 0, 0, 0)
            continue

        # ---------- 탐지 ----------
        t0 = time.time()
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        det = results[0]
        t1 = time.time()

        infer_counter += 1
        if t1 - last_infer_time >= 1.0:
            print(f"[🧠 Inference FPS] {infer_counter} fps")
            infer_counter = 0
            last_infer_time = t1

        # YOLO → DeepSORT 입력
        person_detections = []
        for box in det.boxes:
            cls = int(box.cls[0])
            if cls == target_class_id:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                person_detections.append(([x1, y1, x2 - x1, y2 - y1], conf, TARGET_CLASS))

        tracks = tracker.update_tracks(person_detections, frame=frame)

        # ---------- 타깃 선택(첫 확정 트랙) ----------
        target_box = None
        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f'ID:{track.track_id}', (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            if target_box is None:
                target_box = (x1, y1, x2, y2)

        # ---------- RC 제어값 계산 (send_rc_control) ----------
        lr = 0  # 좌우 평행이동은 0 (원하면 yaw 대신 lr로 바꿀 수 있음)
        fb = 0
        ud = 0
        yaw = 0

        if target_box:
            x1, y1, x2, y2 = target_box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            width = x2 - x1

            # 오차 계산
            err_x = cx - CENTER_X     # +: 오른쪽으로 치우침
            err_y = cy - CENTER_Y     # +: 아래로 치우침 (이미지는 아래가 +)
            err_w = DESIRED_WIDTH - width  # +: 더 가까이(전진) 필요

            # 데드존 적용
            if abs(err_x) < CENTER_MARGIN: err_x = 0
            if abs(err_y) < CENTER_MARGIN: err_y = 0
            if abs(err_w) < 20:            err_w = 0  # 거리 유지 데드존(픽셀)

            # 비례제어 (부호 방향 주의)
            yaw =  int(max(min(Kp_yaw * err_x,  MAX_YAW), -MAX_YAW))     # +면 시계 회전
            ud  = -int(max(min(Kp_ud  * err_y,  MAX_UD ), -MAX_UD ))     # 화면 아래(+err_y)는 드론 하강(+ud), 우리가 원하는 건 상승(-)이므로 부호 반전
            fb  =  int(max(min(Kp_fb  * err_w,  MAX_FB ), -MAX_FB ))     # +면 전진

        else:
            # 타깃 없으면 서서히 감쇠해 정지
            yaw = int(prev_yaw * DECAY)
            ud  = int(prev_ud  * DECAY)
            fb  = int(prev_fb  * DECAY)
            lr  = int(prev_lr  * DECAY)

        # 스무딩(EMA)
        lr  = int(SMOOTH_ALPHA * prev_lr  + (1-SMOOTH_ALPHA) * lr)
        fb  = int(SMOOTH_ALPHA * prev_fb  + (1-SMOOTH_ALPHA) * fb)
        ud  = int(SMOOTH_ALPHA * prev_ud  + (1-SMOOTH_ALPHA) * ud)
        yaw = int(SMOOTH_ALPHA * prev_yaw + (1-SMOOTH_ALPHA) * yaw)

        # RC 명령 전송 (20Hz 내외)
        tello.send_rc_control(lr, fb, ud, yaw)

        # 이전값 업데이트
        prev_lr, prev_fb, prev_ud, prev_yaw = lr, fb, ud, yaw

        # ---------- 디스플레이 ----------
        # 가이드 라인
        cv2.line(frame, (CENTER_X - CENTER_MARGIN, CENTER_Y), (CENTER_X + CENTER_MARGIN, CENTER_Y), (255,255,255), 1)
        cv2.line(frame, (CENTER_X, CENTER_Y - CENTER_MARGIN), (CENTER_X, CENTER_Y + CENTER_MARGIN), (255,255,255), 1)
        cv2.putText(frame, f"RC lr:{lr} fb:{fb} ud:{ud} yaw:{yaw}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Drone View (YOLOv8 + LifoQueue + DeepSORT + RC)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # 루프 템포 (추론 시간이 길면 이 슬립은 거의 무시됨)
        time.sleep(0.02)

finally:
    # 정리
    stop_event.set()
    try:
        reader_thread.join(timeout=1.0)
    except:
        pass

    # 정지 명령 후 착륙
    try:
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.2)
        tello.land()
    except Exception as e:
        print("착륙 중 예외:", e)

    tello.streamoff()
    cv2.destroyAllWindows()

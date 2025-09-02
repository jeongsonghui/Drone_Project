from djitellopy import Tello
import cv2
from ultralytics import YOLO
import threading
import queue
import time
import logging
from logging.handlers import QueueHandler, QueueListener
from collections import deque
import statistics

# =========================
# Logging Setup
# =========================
log_queue = queue.Queue(-1)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("drone.log", mode='a', encoding='utf-8')
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
queue_handler = QueueHandler(log_queue)
listener = QueueListener(log_queue, console_handler, file_handler)

logger = logging.getLogger("DroneLogger")
logger.setLevel(logging.INFO)
logger.addHandler(queue_handler)
listener.start()

# =========================
# Model and Drone Setup
# =========================
MODEL_PATH = "C:/Users/Administrator/Desktop/Drone_Project/models/best4.pt"
TARGET_LABEL = "lifter"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CENTER_X = FRAME_WIDTH // 2
CENTER_Y = FRAME_HEIGHT // 2

# 제어 파라미터 (환경에 맞게 튜닝)
TARGET_BOX_HEIGHT = 150  # 원하는 거리(박스 높이) 타깃
K_YAW = 0.25             # 좌우 회전 비례 게인
K_UP  = 0.22             # 상하 비례 게인
K_FB  = 0.18             # 전/후진 비례 게인
VEL_CLAMP = 35           # 속도 클램프 (안전범위 권장 20~40)
DEADBAND_PX = 14         # 미세 떨림 억제 데드밴드(픽셀)

# EMA(지수평활) 계수
EMA_A = 0.2              # 0.2~0.5 사이 추천

# 유실/스캔/타임아웃
SCAN_YAW = 14            # 유실 시 좌우 스윕 속도
SHORT_LOST_SECS = 3.0    # 최근 유실: 호버
LONG_LOST_SECS  = 15.0   # 장기 유실: 자동 착륙

# 탐지 임계 강화
DET_CONF = 0.50          # (기존 0.45 → 0.50)
DET_IOU  = 0.50          # (기존 0.45 → 0.50)

# 속도 상한 및 변화율 제한
VEL_CLAMP_YAW = 35
VEL_CLAMP_FB  = 24
VEL_CLAMP_UD  = 24
SLEW_DV_MAX   = 10       # 프레임당 속도 변화 최대치

# 속도 저역통과(velocity smoothing)
VLP_A = 0.65             # v_smooth = A*v_prev + (1-A)*v_new

# 미디안 윈도 크기(좌표/높이)
MED_WIN = 5    

# RC 전송 고정 주기(20Hz)
RC_PERIOD = 0.05

# 스레드 종료 Event
stop_event = threading.Event()

# FPS 계수기
io_counter = 0
infer_counter = 0
last_io_time = time.time()
last_infer_time = time.time()

# EMA 상태값(초기값)
ema_cx = CENTER_X
ema_cy = CENTER_Y
ema_h  = TARGET_BOX_HEIGHT

# 추적 상태(이전 박스, 미디안 버퍼, 속도 스무딩)
prev_box = None  # (x1,y1,x2,y2)

med_cx = deque(maxlen=MED_WIN)
med_cy = deque(maxlen=MED_WIN)
med_h  = deque(maxlen=MED_WIN)

vprev_yaw = 0
vprev_fb  = 0
vprev_ud  = 0

# 마지막 타깃 시각
last_seen_time = time.time()

# 프레임 큐(지연 최소화)
frame_queue = queue.Queue(maxsize=3)

# 모델 로드
model = YOLO(MODEL_PATH)
# 선택: GPU 사용 가능 시 가속
try:
    import torch
    if torch.cuda.is_available():
        model.to('cuda')
        logger.info("모델을 CUDA로 이동했습니다.")
except Exception as e:
    logger.info(f"CUDA 가속 사용 불가 또는 무시: {e}")

# 타깃 클래스 id 탐색(있을 시 classes 필터에 사용)
def get_target_class_id():
    try:
        # model.names: {id: "name"} 형태
        for cid, name in model.names.items():
            if name == TARGET_LABEL:
                return int(cid)
    except Exception:
        pass
    return None

target_class_id = get_target_class_id()
if target_class_id is not None:
    logger.info(f"타깃 클래스 '{TARGET_LABEL}' id: {target_class_id}")
else:
    logger.warning(f"타깃 라벨 '{TARGET_LABEL}' 을(를) model.names에서 찾지 못했습니다. 전체 클래스에서 탐지합니다.")

# 드론 연결
tello = Tello()
tello.connect()
logger.info(f"배터리: {tello.get_battery()}%")
tello.streamon()
frame_read = tello.get_frame_read()

# =========================
# 유틸
# =========================
def clamp(v, low, high):
    return max(min(v, high), low)

def iou_xyxy(a, b):
    """a,b = (x1,y1,x2,y2)"""
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aarea = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    barea = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    denom = aarea + barea - inter
    return inter / denom if denom > 0 else 0.0

def slew_limit(v_prev, v_new, dv_max):
    """프레임당 속도 변화율 제한"""
    delta = v_new - v_prev
    if delta > dv_max:
        return v_prev + dv_max
    if delta < -dv_max:
        return v_prev - dv_max
    return v_new

# =========================
# Frame Fetch Thread
# =========================
def fetch_frames():
    global io_counter, last_io_time
    while not stop_event.is_set():
        frame = frame_read.frame
        if frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            # 가장 오래된 프레임 버리고 최신 프레임 투입(지연 최소화)
            try:
                frame_queue.get_nowait()
                frame_queue.put_nowait(frame)
            except queue.Empty:
                pass

        io_counter += 1
        now = time.time()
        if now - last_io_time >= 1.0:
            logger.info(f"[IO FPS] {io_counter} fps")
            io_counter = 0
            last_io_time = now

        time.sleep(0.005)

# =========================
# Detection / Control Thread
# =========================
def process_frames():
    global infer_counter, last_infer_time
    global ema_cx, ema_cy, ema_h, last_seen_time
    global prev_box, med_cx, med_cy, med_h
    global vprev_yaw, vprev_fb, vprev_ud

    while not stop_event.is_set():
        t_loop = time.time()

        # 키 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_event.set()
            break
        elif key == ord('h'):
            try:
                tello.send_rc_control(0, 0, 0, 0)
            except Exception:
                pass
        elif key == ord('l'):
            try:
                tello.send_rc_control(0, 0, 0, 0)
                tello.land()
            except Exception:
                pass
            stop_event.set()
            break

        if frame_queue.empty():
            dt = time.time() - t_loop
            rem = RC_PERIOD - dt
            if rem > 0:
                time.sleep(rem)
            continue

        frame = frame_queue.get()

        # --- 추론: 모든 클래스 감지 (lifter + person 동시) ---
        results = model.predict(frame, conf=DET_CONF, iou=DET_IOU, verbose=False, persist=True)[0]

        infer_counter += 1
        now = time.time()
        if now - last_infer_time >= 1.0:
            logger.info(f"[Inference FPS] {infer_counter} fps")
            infer_counter = 0
            last_infer_time = now

        yaw_velocity = 0
        up_down_velocity = 0
        forward_backward_velocity = 0
        target_found = False

        boxes_raw = results.boxes.data.tolist() if results.boxes is not None else []
        logger.info(f"[탐지결과] 박스 {len(boxes_raw)}개")

        # --- 사람: HUD 빨간 박스만 ---
        person_count = 0
        for b in boxes_raw:
            x1, y1, x2, y2, conf, cls = b
            cls = int(cls)
            label = model.names.get(cls, str(cls)).lower()
            if label in ("person", "human"):
                person_count += 1
                p1 = (int(x1), int(y1)); p2 = (int(x2), int(y2))
                cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)
                cv2.putText(frame, f"person ({float(conf):.2f})",
                            (p1[0], max(0, p1[1]-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if person_count:
            cv2.putText(frame, f"Persons: {person_count}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- lifter 후보만 모으기 ---
        lifter_candidates = []
        for b in boxes_raw:
            x1, y1, x2, y2, conf, cls = b
            label = model.names.get(int(cls), str(cls))
            if label == TARGET_LABEL:
                lifter_candidates.append((x1, y1, x2, y2, float(conf)))

        # --- 최적 lifter 선택: conf + 0.2*centrality + 0.4*IoU(prev) ---
        best = None
        best_score = -1e9
        for (x1, y1, x2, y2, conf) in lifter_candidates:
            cx = (x1 + x2) / 2.0
            centrality = 1.0 - abs(cx - CENTER_X) / CENTER_X
            iou_prev = iou_xyxy(prev_box, (x1, y1, x2, y2))
            score = conf + 0.2 * centrality + 0.4 * iou_prev
            if score > best_score:
                best_score = score
                best = (x1, y1, x2, y2, conf)

        # --- lifter 추적/제어 ---
        if best is not None:
            x1, y1, x2, y2, confidence = best
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            box_height = int(y2 - y1)

            # 미디안 버퍼 업데이트
            med_cx.append(cx); med_cy.append(cy); med_h.append(box_height)
            cx_m = int(statistics.median(med_cx))
            cy_m = int(statistics.median(med_cy))
            h_m  = int(statistics.median(med_h))

            # EMA(미디안 → EMA)로 저역통과
            ema_cx = int(EMA_A * cx_m + (1 - EMA_A) * ema_cx)
            ema_cy = int(EMA_A * cy_m + (1 - EMA_A) * ema_cy)
            ema_h  = int(EMA_A * h_m  + (1 - EMA_A) * ema_h)

            # 제어 에러
            dx = ema_cx - CENTER_X
            dy = ema_cy - CENTER_Y
            size_err = TARGET_BOX_HEIGHT - ema_h

            # 데드밴드
            if abs(dx) < DEADBAND_PX: dx = 0
            if abs(dy) < DEADBAND_PX: dy = 0

            # P 제어 (필요시 Ki 추가 가능)
            yaw_cmd = K_YAW * dx
            ud_cmd  = -K_UP * dy
            fb_cmd  = K_FB * size_err

            # 1) 속도 클램프(축별)
            yaw_cmd = int(clamp(yaw_cmd, -VEL_CLAMP_YAW, VEL_CLAMP_YAW))
            ud_cmd  = int(clamp(ud_cmd,  -VEL_CLAMP_UD,  VEL_CLAMP_UD))
            fb_cmd  = int(clamp(fb_cmd,  -VEL_CLAMP_FB,  VEL_CLAMP_FB))

            # 2) 속도 저역통과
            yaw_cmd = int(VLP_A * vprev_yaw + (1 - VLP_A) * yaw_cmd)
            ud_cmd  = int(VLP_A * vprev_ud  + (1 - VLP_A) * ud_cmd)
            fb_cmd  = int(VLP_A * vprev_fb  + (1 - VLP_A) * fb_cmd)

            # 3) 변화율 제한(slew)
            yaw_cmd = int(slew_limit(vprev_yaw, yaw_cmd, SLEW_DV_MAX))
            ud_cmd  = int(slew_limit(vprev_ud,  ud_cmd,  SLEW_DV_MAX))
            fb_cmd  = int(slew_limit(vprev_fb,  fb_cmd,  SLEW_DV_MAX))

            # 적용 및 상태 갱신
            yaw_velocity, up_down_velocity, forward_backward_velocity = yaw_cmd, ud_cmd, fb_cmd
            vprev_yaw, vprev_ud, vprev_fb = yaw_cmd, ud_cmd, fb_cmd
            prev_box = (x1, y1, x2, y2)

            last_seen_time = time.time()
            target_found = True

            # HUD: lifter 초록 박스 + 상태
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{TARGET_LABEL} ({confidence:.2f})",
                        (int(x1), max(0, int(y1)-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.circle(frame, (ema_cx, ema_cy), 4, (0, 255, 0), -1)
            cv2.putText(frame, f"EMA(cx,cy,h)=({ema_cx},{ema_cy},{ema_h})", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
            cv2.putText(frame, f"RC FB:{forward_backward_velocity} UD:{up_down_velocity} Yaw:{yaw_velocity}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

            logger.info(f"[추적] lifter @ ({ema_cx},{ema_cy}) h={ema_h} | RC FB {fb_cmd}, UD {ud_cmd}, Yaw {yaw_cmd}")

        # --- lifter 미탐: 호버 → 완만 스캔 → 타임아웃 착륙 ---
        if not target_found:
            miss_sec = time.time() - last_seen_time
            if miss_sec < SHORT_LOST_SECS:
                yaw_velocity = up_down_velocity = forward_backward_velocity = 0
                # 상태 서서히 0으로
                vprev_yaw = int(VLP_A * vprev_yaw)
                vprev_ud  = int(VLP_A * vprev_ud)
                vprev_fb  = int(VLP_A * vprev_fb)
                logger.info("리프터 미탐: 최근 유실 → 호버")
            else:
                yaw_velocity = SCAN_YAW
                up_down_velocity = forward_backward_velocity = 0
                vprev_yaw = yaw_velocity; vprev_ud = 0; vprev_fb = 0
                logger.info("리프터 미탐: 스캔 중(좌우 회전)")

            if miss_sec > LONG_LOST_SECS:
                logger.warning("타깃 장기 유실 → 안전 착륙")
                try:
                    tello.send_rc_control(0, 0, 0, 0)
                    tello.land()
                except Exception:
                    pass
                stop_event.set()
                break

        # --- RC 전송 ---
        try:
            tello.send_rc_control(0, forward_backward_velocity, up_down_velocity, yaw_velocity)
        except Exception as e:
            logger.warning(f"[명령 전송 오류] {e}")

        # --- HUD 공통 ---
        cv2.putText(frame, f"Lost: {int(time.time() - last_seen_time)}s", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

        cv2.imshow("📷 Tello Camera", frame)

        # 주기 맞추기
        dt = time.time() - t_loop
        rem = RC_PERIOD - dt
        if rem > 0:
            time.sleep(rem)

    # 종료 루틴
    try:
        tello.send_rc_control(0, 0, 0, 0)
        tello.land()
    except Exception:
        pass

# =========================
# Main
# =========================
try:
    tello.takeoff()
    time.sleep(2)

    fetch_thread = threading.Thread(target=fetch_frames, daemon=True)
    process_thread = threading.Thread(target=process_frames, daemon=False)

    fetch_thread.start()
    process_thread.start()
    process_thread.join()

except KeyboardInterrupt:
    logger.info("Ctrl+C에 의한 종료 요청 감지됨")
    stop_event.set()
    try:
        tello.send_rc_control(0, 0, 0, 0)
        tello.land()
    except Exception:
        pass

finally:
    stop_event.set()
    try:
        tello.send_rc_control(0, 0, 0, 0)
    except Exception:
        pass
    try:
        tello.streamoff()
    except Exception:
        pass
    cv2.destroyAllWindows()
    listener.stop()
    logger.info("모든 리소스 종료 완료")


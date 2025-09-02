from djitellopy import Tello
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
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
PERSON_LABELS = "person"

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
    global ema_cx, ema_cy, ema_h, last_seen_time
    global prev_box, med_cx, med_cy, med_h
    global vprev_yaw, vprev_ud, vprev_fb
    global target_track_id

    while not stop_event.is_set():
        t0 = time.time()

        # 키 입력
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_event.set(); break
        elif key == ord('h'):
            try: tello.send_rc_control(0,0,0,0)
            except: pass
        elif key == ord('l'):
            try: tello.send_rc_control(0,0,0,0); tello.land()
            except: pass
            stop_event.set(); break

        if frame_queue.empty():
            rem = RC_PERIOD - (time.time() - t0)
            if rem > 0: time.sleep(rem)
            continue

        frame = frame_queue.get()

        # ---------- YOLO 추론 ----------
        results = model.predict(frame, conf=DET_CONF, iou=DET_IOU, verbose=False, persist=True)[0]
        boxes = results.boxes
        detections = []  # DeepSORT 입력: (tlwh, conf, class_name)
        persons_xyxy = []  # HUD용 사람 박스

        if boxes is not None:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy()
            clss  = boxes.cls.detach().cpu().numpy().astype(int)

            for (x1,y1,x2,y2), conf, cls in zip(xyxy, confs, clss):
                w, h = x2 - x1, y2 - y1
                label = model.names.get(cls, str(cls))
                if label == TARGET_LABEL:
                    detections.append(([float(x1), float(y1), float(w), float(h)], float(conf), label))
                elif label.lower() in PERSON_LABELS:
                    persons_xyxy.append((int(x1), int(y1), int(x2), int(y2), float(conf)))

        # ---------- DeepSORT 업데이트 (리프터만) ----------
        tracks = tracker.update_tracks(detections, frame=frame)

        # 사람 HUD (빨간 박스)
        for (x1,y1,x2,y2,conf) in persons_xyxy:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(frame, f"person ({conf:.2f})", (x1, max(0,y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        if persons_xyxy:
            cv2.putText(frame, f"Persons: {len(persons_xyxy)}", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # ---------- 타깃 트랙 선택/유지 ----------
        target_box = None
        candidates = []  # (score, track_id, (l,t,r,b), conf)

        for trk in tracks:
            if not trk.is_confirmed(): 
                continue
            l,t,r,b = map(int, trk.to_ltrb())
            # 일부 구현에서 class/conf 저장 못할 수 있어 fallback
            trk_label = (getattr(trk, "det_class", None) or trk.get_det_class() or TARGET_LABEL)
            trk_conf  = (getattr(trk, "det_conf", None)  or trk.get_det_conf()  or 0.0)

            if trk_label == TARGET_LABEL:
                cx = (l + r) / 2.0
                centrality = 1.0 - abs(cx - CENTER_X) / CENTER_X
                score = float(trk_conf) + 0.2 * centrality
                candidates.append((score, trk.track_id, (l,t,r,b), float(trk_conf)))

        # 1) 이전 타깃 유지
        if target_track_id is not None:
            for _, tid, box, _ in candidates:
                if tid == target_track_id:
                    target_box = box
                    break

        # 2) 없으면 새로 선택
        if target_box is None and candidates:
            candidates.sort(reverse=True)   # score 내림차순
            _, target_track_id, target_box, _ = candidates[0]

        # ---------- 제어(안정화 포함) ----------
        yaw_cmd = ud_cmd = fb_cmd = 0
        if target_box:
            x1,y1,x2,y2 = target_box
            cx = (x1 + x2)//2
            cy = (y1 + y2)//2
            h  = (y2 - y1)

            # 미디안 → EMA
            med_cx.append(cx); med_cy.append(cy); med_h.append(h)
            cx_m = int(statistics.median(med_cx)) if len(med_cx)>0 else cx
            cy_m = int(statistics.median(med_cy)) if len(med_cy)>0 else cy
            h_m  = int(statistics.median(med_h))  if len(med_h)>0  else h

            ema_cx = int(EMA_A * cx_m + (1 - EMA_A) * ema_cx)
            ema_cy = int(EMA_A * cy_m + (1 - EMA_A) * ema_cy)
            ema_h  = int(EMA_A * h_m  + (1 - EMA_A) * ema_h)

            dx = ema_cx - CENTER_X
            dy = ema_cy - CENTER_Y
            size_err = TARGET_BOX_HEIGHT - ema_h

            if abs(dx) < DEADBAND_PX: dx = 0
            if abs(dy) < DEADBAND_PX: dy = 0

            yaw_cmd = int(clamp(K_YAW * dx, -VEL_CLAMP_YAW, VEL_CLAMP_YAW))
            ud_cmd  = int(clamp(-K_UP * dy,  -VEL_CLAMP_UD,  VEL_CLAMP_UD))
            fb_cmd  = int(clamp(K_FB * size_err, -VEL_CLAMP_FB, VEL_CLAMP_FB))

            # 속도 저역통과
            yaw_cmd = int(VLP_A * vprev_yaw + (1 - VLP_A) * yaw_cmd)
            ud_cmd  = int(VLP_A * vprev_ud  + (1 - VLP_A) * ud_cmd)
            fb_cmd  = int(VLP_A * vprev_fb  + (1 - VLP_A) * fb_cmd)

            # 변화율 제한
            yaw_cmd = int(slew_limit(vprev_yaw, yaw_cmd, SLEW_DV_MAX))
            ud_cmd  = int(slew_limit(vprev_ud,  ud_cmd,  SLEW_DV_MAX))
            fb_cmd  = int(slew_limit(vprev_fb,  fb_cmd,  SLEW_DV_MAX))

            # 상태 갱신
            vprev_yaw, vprev_ud, vprev_fb = yaw_cmd, ud_cmd, fb_cmd
            prev_box = (x1, y1, x2, y2)
            last_seen_time = time.time()

            # HUD
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{TARGET_LABEL} ID:{target_track_id}",
                        (x1, max(0,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.circle(frame, (ema_cx, ema_cy), 4, (0,255,0), -1)
            cv2.putText(frame, f"EMA({ema_cx},{ema_cy}) h={ema_h}", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2)
        else:
            # 유실 처리: 호버 → 스캔 → 타임아웃
            miss = time.time() - last_seen_time
            if miss < SHORT_LOST_SECS:
                yaw_cmd = ud_cmd = fb_cmd = 0
                vprev_yaw = int(VLP_A * vprev_yaw)
                vprev_ud  = int(VLP_A * vprev_ud)
                vprev_fb  = int(VLP_A * vprev_fb)
            else:
                yaw_cmd = SCAN_YAW
                ud_cmd = fb_cmd = 0
                vprev_yaw, vprev_ud, vprev_fb = yaw_cmd, 0, 0

            if miss > LONG_LOST_SECS:
                try: tello.send_rc_control(0,0,0,0); tello.land()
                except: pass
                stop_event.set(); break

        # RC 전송
        try:
            tello.send_rc_control(0, fb_cmd, ud_cmd, yaw_cmd)
        except Exception as e:
            # 드론이 순간 응답 못할 경우도 있으니 경고만
            # logger.warning(f"[RC 오류] {e}")
            pass

        # 공통 HUD
        cv2.putText(frame, f"Lost: {int(time.time() - last_seen_time)}s", (10,75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2)
        cv2.imshow("📷 Tello Camera", frame)

        # 주기 맞추기
        rem = RC_PERIOD - (time.time() - t0)
        if rem > 0:
            time.sleep(rem)

    # 종료 루틴
    try:
        tello.send_rc_control(0,0,0,0)
        tello.land()
    except:
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


import cv2
import time
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ================== 설정 ==================
TARGET_CLASS   = 'person'
CONF_THRESHOLD = 0.4
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480
SHOW_GUIDE = True
# =========================================

# ======= YOLOv8n 모델 로드 =======
model = YOLO("yolov8n.pt")  # ultralytics 설치 필요: pip install ultralytics

# 모델 클래스 이름 확인
names = model.names
if TARGET_CLASS not in names.values():
    raise ValueError(f"'{TARGET_CLASS}' not in model class names: {names}")
# 클래스 ID 가져오기
target_class_id = [k for k, v in names.items() if v == TARGET_CLASS][0]

# ======= Deep SORT 초기화 =======
tracker = DeepSort(max_age=30)

# ======= 웹캠 연결 =======
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

def pick_target_center_nearest(tracks, frame_w, frame_h):
    cx, cy = frame_w // 2, frame_h // 2
    best = None
    best_dist2 = 1e18
    for tr in tracks:
        if not tr.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, tr.to_ltrb())
        tx = (x1 + x2) // 2
        ty = (y1 + y2) // 2
        dist2 = (tx - cx) ** 2 + (ty - cy) ** 2
        if dist2 < best_dist2:
            best_dist2 = dist2
            best = (tr, (x1, y1, x2, y2), (tx, ty))
    return best

prev_time = time.time()

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("카메라 프레임을 가져올 수 없습니다.")
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # ====== YOLOv8 탐지 ======
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        detections = results[0]

        person_detections = []
        for box in detections.boxes:
            cls = int(box.cls[0])
            if cls == target_class_id:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                w, h = x2 - x1, y2 - y1
                person_detections.append(([x1, y1, w, h], conf, TARGET_CLASS))

        # ====== 추적 ======
        tracks = tracker.update_tracks(person_detections, frame=frame)

        # ====== 타깃 선택 ======
        picked = pick_target_center_nearest(tracks, FRAME_WIDTH, FRAME_HEIGHT)

        # ====== 시각화 ======
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, tr.to_ltrb())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f'ID: {tr.track_id}', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if picked is not None:
            _, (x1, y1, x2, y2), (tx, ty) = picked
            cv2.circle(frame, (tx, ty), 4, (0,255,255), -1)
            cv2.putText(frame, "TARGET", (x1, max(0, y1-35)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        if SHOW_GUIDE:
            cv2.circle(frame, (FRAME_WIDTH//2, FRAME_HEIGHT//2), 5, (255,255,255), -1)

        # FPS 표시
        now = time.time()
        fps = 1.0 / max(1e-6, (now - prev_time))
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Webcam Person Tracking (YOLOv8n + DeepSORT)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

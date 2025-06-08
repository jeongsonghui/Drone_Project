import cv2
import torch
from djitellopy import Tello
from deep_sort_realtime.deepsort_tracker import DeepSort

# =============== 설정 ================
TARGET_CLASS = 'person'  # 추적할 객체 클래스
CONF_THRESHOLD = 0.4
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CENTER_MARGIN = 50  # 중심에서 얼마나 벗어나야 움직일지
# ====================================

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = CONF_THRESHOLD
target_class_id = model.names.index(TARGET_CLASS)

# Deep SORT 초기화
tracker = DeepSort(max_age=30)

# DJI Tello 연결
tello = Tello()
tello.connect()
print("배터리:", tello.get_battery(), "%")
tello.streamon()

# 비행 시작
tello.takeoff()

# 프레임 반복
try:
    while True:
        frame = tello.get_frame_read().frame
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # YOLO 탐지
        results = model(frame)
        detections = results.pred[0]

        person_detections = []
        for *xyxy, conf, cls in detections:
            if int(cls) == target_class_id:
                x1, y1, x2, y2 = map(int, xyxy)
                conf = float(conf)
                person_detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

        tracks = tracker.update_tracks(person_detections, frame=frame)
        target_box = None

        # 추적 중인 객체 시각화 및 대상 추출
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 첫 번째 객체를 추적 대상으로 설정
            if target_box is None:
                target_box = (x1, y1, x2, y2)

        # 드론 제어 로직
        if target_box:
            x1, y1, x2, y2 = target_box
            obj_center_x = (x1 + x2) // 2
            obj_center_y = (y1 + y2) // 2

            frame_center_x = FRAME_WIDTH // 2
            frame_center_y = FRAME_HEIGHT // 2

            # 좌우 이동
            if obj_center_x < frame_center_x - CENTER_MARGIN:
                tello.move_left(20)
            elif obj_center_x > frame_center_x + CENTER_MARGIN:
                tello.move_right(20)

            # 상하 이동
            if obj_center_y < frame_center_y - CENTER_MARGIN:
                tello.move_up(20)
            elif obj_center_y > frame_center_y + CENTER_MARGIN:
                tello.move_down(20)

            # 앞으로/뒤로 이동 (객체 크기로 거리 추정 가능)
            width = x2 - x1
            if width < 100:
                tello.move_forward(20)
            elif width > 200:
                tello.move_back(20)

        cv2.imshow("Drone View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    tello.land()
    tello.streamoff()
    cv2.destroyAllWindows()

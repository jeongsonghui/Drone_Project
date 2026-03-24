from djitellopy import Tello
import cv2
import time

# Tello 객체 생성
tello = Tello()
tello.connect()

# 카메라 스트리밍 활성화
tello.streamon()

# 이륙
tello.takeoff()
time.sleep(2)

# 카메라 방향을 패드를 향하게 조정 (90도 회전해서 아래로 향하도록 설정 가능)
tello.rotate_clockwise(90)  

# 그리드 설정 (20cm 간격)
grid_size = 20  # 각 패드 간 거리
positions = {
    1: (0, 0), 2: (1, 0), 3: (2, 0),
    4: (0, 1), 5: (1, 1), 6: (2, 1),
    7: (0, 2), 8: (1, 2), 9: (2, 2)
}

# 현재 패드 확인 (방향 전환 시)
def capture_frame():
    frame = tello.get_frame_read().frame
    cv2.imshow("Tello Camera", frame)
    cv2.waitKey(1)  # 이미지 갱신

# 이동 함수
def move_to(target_id, current_pos, last_direction):
    target_pos = positions[target_id]
    dx = (target_pos[0] - current_pos[0]) * grid_size  # x 방향 이동 거리
    dy = (target_pos[1] - current_pos[1]) * grid_size  # y 방향 이동 거리

    current_direction = None

    # x축 이동
    if dx > 0:
        current_direction = 'right'
        tello.move_right(abs(dx))
    elif dx < 0:
        current_direction = 'left'
        tello.move_left(abs(dx))

    # y축 이동
    if dy > 0:
        current_direction = 'up'
        tello.move_up(abs(dy))
    elif dy < 0:
        current_direction = 'down'
        tello.move_down(abs(dy))

    time.sleep(1)  # 이동 후 안정화 대기

    # 방향이 바뀌었을 때만 카메라 확인
    if current_direction != last_direction:
        capture_frame()

    return target_pos, current_direction  # 현재 위치 및 이동 방향 업데이트

# 이동 경로 설정 (예: 1 → 5 → 9)
path = [1, 5, 9]
current_position = positions[path[0]]
last_direction = None

# 경로 따라 이동 (방향 전환 시만 카메라 확인)
for target in path[1:]:
    current_position, last_direction = move_to(target, current_position, last_direction)
    time.sleep(1)

# 착륙
tello.land()

# 스트리밍 종료
cv2.destroyAllWindows()
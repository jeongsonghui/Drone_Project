from djitellopy import Tello
import time

# Tello 객체 생성
tello = Tello()
tello.connect()

# 이륙
tello.takeoff()
time.sleep(2)  # 안정화를 위해 대기

grid_size = 30  # 각 그리드 크기 (25cm)
positions = {
    1: (0, 0), 2: (1, 0), 3: (2, 0),
    4: (0, 1), 5: (1, 1), 6: (2, 1),
    7: (0, 2), 8: (1, 2), 9: (2, 2)
}

def move_to(target_id, current_pos):
    target_pos = positions[target_id]
    dx = (target_pos[0] - current_pos[0]) * grid_size
    dy = (target_pos[1] - current_pos[1]) * grid_size

    # x축 이동
    if dx > 0:
        tello.send_rc_control(40, 0, 0, 0)
    elif dx < 0:
        tello.send_rc_control(-40, 0, 0, 0)

    time.sleep(2)  # 이동 후 대기
    tello.send_rc_control(0, 0, 0, 0)  # 정지

    # y축 이동
    if dy > 0:
        tello.send_rc_control(0, 40, 0, 0)
    elif dy < 0:
        tello.send_rc_control(0, 40, 0, 0)

    time.sleep(2)
    tello.send_rc_control(0, 0, 0, 0)

    return target_pos

# 경로 정의 
path = [1, 4, 5, 6]
current_position = positions[path[0]]

for target in path[1:]:
    current_position = move_to(target, current_position)
    time.sleep(1)

# 착륙
tello.land()

from djitellopy import Tello
import cv2
import time

# Tello 객체 생성
tello = Tello()
tello.connect()

# 비디오 스트리밍 시작
tello.streamon()

# 이륙 후 안정화
tello.takeoff()
time.sleep(1)

# move_down(20) 대신 send_rc_control 사용
tello.send_rc_control(0, 0, -20, 0)  # 약간 하강
time.sleep(2)
tello.send_rc_control(0, 0, 0, 0)  # 정지
time.sleep(1)

# 이동 그리드 설정
grid_size = 20
positions = {
    1: (0, 0), 2: (1, 0), 3: (2, 0),
    4: (0, 1), 5: (1, 1), 6: (2, 1),
    7: (0, 2), 8: (1, 2), 9: (2, 2)
}

def move_to(target_id, current_pos):
    target_pos = positions[target_id]
    dx = (target_pos[0] - current_pos[0]) * grid_size
    dy = (target_pos[1] - current_pos[1]) * grid_size

    # 카메라를 아래로 기울여 바닥 확인
    tello.rotate_clockwise(90)
    time.sleep(2)  # 회전 후 안정화 대기
    
    # 실시간 영상 출력
    show_video()

    # x축 이동 (속도 조정)
    if dx > 0:
        tello.send_rc_control(15, 0, 0, 0)  # 오른쪽 이동
    elif dx < 0:
        tello.send_rc_control(-15, 0, 0, 0)  # 왼쪽 이동

    time.sleep(2)  # 이동 후 대기
    tello.send_rc_control(0, 0, 0, 0)  # 정지
    time.sleep(1)

    # y축 이동 (속도 조정)
    if dy > 0:
        tello.send_rc_control(0, 15, 0, 0)  # 앞으로 이동
    elif dy < 0:
        tello.send_rc_control(0, -15, 0, 0)  # 뒤로 이동

    time.sleep(2)
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(1)

    # 카메라 다시 정면으로
    tello.rotate_counter_clockwise(90)
    time.sleep(2)

    return target_pos

def show_video():
    """ Tello의 실시간 카메라 영상을 표시하는 함수 """
    frame_read = tello.get_frame_read()

    for _ in range(30):  # 약 1초간 영상 표시 (30프레임)
        frame = frame_read.frame
        frame = cv2.resize(frame, (480, 360))  # 화면 크기 조정
        cv2.imshow("Tello Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'를 누르면 종료
            break

# 경로 정의
path = [1, 4, 5, 6]
current_position = positions[path[0]]

for target in path[1:]:
    current_position = move_to(target, current_position)
    time.sleep(1)

# 착륙
tello.land()

# 비디오 스트리밍 종료
tello.streamoff()
cv2.destroyAllWindows()

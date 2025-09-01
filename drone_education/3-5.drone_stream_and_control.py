from djitellopy import Tello
import keyboard
from time import sleep
import cv2

# 드론 초기화 및 연결
drone = Tello()
drone.connect()
print(f"Battery: {drone.get_battery()}%")

# 영상 스트리밍 시작
drone.streamon()
frame_reader = drone.get_frame_read()

# 이륙
print("🔼 UP 키로 이륙, 🔽 DOWN 키로 착륙")
print("w/s: 앞/뒤 | a/d: 좌/우 | k/l: 위/아래 | i/o: 회전")
drone.send_rc_control(0, 0, 0, 0)

while True:
    # ================== 화면 표시 ==================
    frame = frame_reader.frame
    if frame is not None:
        cv2.imshow("Tello Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # q 누르면 종료
        drone.land()
        break

    # ================== 키보드 조작 ==================
    if keyboard.is_pressed("UP"):
        drone.takeoff()
        sleep(1)
    elif keyboard.is_pressed("DOWN"):
        drone.land()
        sleep(1)

    elif keyboard.is_pressed("w"):
        drone.send_rc_control(0, 30, 0, 0)
    elif keyboard.is_pressed("s"):
        drone.send_rc_control(0, -30, 0, 0)
    elif keyboard.is_pressed("a"):
        drone.send_rc_control(-30, 0, 0, 0)
    elif keyboard.is_pressed("d"):
        drone.send_rc_control(30, 0, 0, 0)
    elif keyboard.is_pressed("k"):
        drone.send_rc_control(0, 0, 30, 0)
    elif keyboard.is_pressed("l"):
        drone.send_rc_control(0, 0, -30, 0)
    elif keyboard.is_pressed("i"):
        drone.send_rc_control(0, 0, 0, -30)
    elif keyboard.is_pressed("o"):
        drone.send_rc_control(0, 0, 0, 30)
    else:
        drone.send_rc_control(0, 0, 0, 0)

    sleep(0.1)

# 종료 처리
drone.streamoff()
cv2.destroyAllWindows()

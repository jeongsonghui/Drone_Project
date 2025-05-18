import socket
import time
from djitellopy import Tello

# 드론 객체 생성 및 연결
tello = Tello()
tello.connect()

# 배터리 상태 확인
print(f"Battery: {tello.get_battery()}%")

# 드론 이륙
tello.takeoff()
time.sleep(2)  # 안정화 대기

# 앞으로 50cm 이동
tello.move_forward(50)
time.sleep(2)  # 안정화 대기

# 드론 착륙
tello.land()

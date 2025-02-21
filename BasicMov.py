from time import sleep
from djitellopy import Tello  # Tello 드론을 제어하는 라이브러리

# 1️⃣ 드론 객체 생성
drone = Tello()

# 2️⃣ 드론과 연결
drone.connect()

# 3️⃣ 배터리 상태 출력
print(drone.get_battery())  # 배터리 퍼센트 출력

# 4️⃣ 이륙 (takeoff)
drone.takeoff()

# 5️⃣ 전진 (속도 50으로 2초 동안)
drone.send_rc_control(0, 50, 0, 0)  
sleep(2)

# 6️⃣ 정지 (모든 움직임 중지)
drone.send_rc_control(0, 0, 0, 0)

# 7️⃣ 착륙 (land)
drone.land()

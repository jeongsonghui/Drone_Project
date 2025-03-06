from djitellopy import Tello
import keyboard
import time

# 드론 객체 생성 및 연결
tello = Tello()
tello.connect()

# 배터리 상태 확인
battery = tello.get_battery()
print(f"🔋 배터리 상태: {battery}%")

if battery < 20:
    print("⚠️ 배터리가 부족합니다! 충전 후 다시 시도하세요.")
    tello.end()
    exit()

# 드론을 명령 모드로 전환 후 대기
tello.send_command_with_return("command")
time.sleep(1)  # 안정성을 위한 짧은 대기 시간

# 이륙
print("🚁 이륙 준비...")
tello.takeoff()
time.sleep(3)  # 안정성을 위한 짧은 대기 시간

# 속도 및 설정값
SPEED = 10  # 이동 속도 (cm/s)
VERTICAL_SPEED = 5  # 상승/하강 속도 (cm/s)
DROP_HEIGHT = 50  # 하강 높이 (cm)

print("🎮 조종 시작: W/A/S/D (앞뒤/좌우), ↑↓ (상승/하강), 'r' (수직 하강 후 복귀), 'q' (착륙 후 종료)")

try:
    while True:
        start_time = time.time()  # 시작 시간 기록

        # 이동 방향 (앞뒤, 좌우)
        if keyboard.is_pressed('w'):
            tello.move_forward(SPEED)
            print("⬆️ 앞쪽으로 이동")
        elif keyboard.is_pressed('s'):
            tello.move_backward(SPEED)
            print("⬇️ 뒤쪽으로 이동")

        elif keyboard.is_pressed('a'):
            tello.move_left(SPEED)
            print("⬅️ 왼쪽으로 이동")
        elif keyboard.is_pressed('d'):
            tello.move_right(SPEED)
            print("➡️ 오른쪽으로 이동")

        # 상승/하강
        if keyboard.is_pressed('up'):
            tello.move_up(VERTICAL_SPEED)
            print("🔼 상승 중")
        elif keyboard.is_pressed('down'):
            tello.move_down(VERTICAL_SPEED)
            print("🔽 하강 중")

        # 특정 키 ('r')를 눌렀을 때, 수직 하강 후 복귀
        elif keyboard.is_pressed('r'):
            print("🔽 하강 시작...")
            tello.move_down(DROP_HEIGHT)
            time.sleep(1)  # 잠시 대기
            print("🔼 다시 상승...")
            tello.move_up(DROP_HEIGHT)

        # ESC 키를 누르면 착륙 후 종료
        elif keyboard.is_pressed('q'):
            print("🛬 드론 착륙 중...")
            tello.land()
            break

        # 실시간 반영을 위한 간격 최소화
        elapsed_time = time.time() - start_time
        if elapsed_time < 0.1:
            time.sleep(0.1 - elapsed_time)  # 최소 0.1초 간격으로 조정

except KeyboardInterrupt:
    print("🚨 인터럽트 감지, 드론 착륙 중...")
    tello.land()

# 드론 종료
tello.end()
print("✅ 프로그램 종료")

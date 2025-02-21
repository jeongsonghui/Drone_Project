from time import sleep
from djitellopy import Tello  # Tello 드론을 제어하는 라이브러리
import cv2
import keyboard
import threading

# 드론 비디오 스트리밍을 별도의 스레드에서 처리
def video_stream_thread():
    while True:
        image = drone.get_frame_read().frame
        image = cv2.resize(image, (500, 460))
        cv2.imshow("Image", image)
        cv2.waitKey(1)

def getInput():
    left_right, front_back, up_down, clock_counter = 0, 0, 0, 0

    if keyboard.is_pressed("a"):  # 'a' 키가 눌리면 왼쪽으로 이동
        left_right = -20
    if keyboard.is_pressed("d"):  # 'd' 키가 눌리면 오른쪽으로 이동
        left_right = 20
    if keyboard.is_pressed("w"):  # 'w' 키가 눌리면 앞쪽으로 이동
        front_back = 20
    if keyboard.is_pressed("s"):  # 's' 키가 눌리면 뒤쪽으로 이동
        front_back = -20

    if keyboard.is_pressed("k"):  # 'k' 키가 눌리면 위로 이동
        up_down = 20
    if keyboard.is_pressed("l"):  # 'l' 키가 눌리면 아래로 이동
        up_down = -20
    if keyboard.is_pressed("i"):  # 'i' 키가 눌리면 시계방향으로 회전
        clock_counter = -20
    if keyboard.is_pressed("o"):  # 'o' 키가 눌리면 반시계방향으로 회전
        clock_counter = 20

    if keyboard.is_pressed("UP"):  # 'UP' 키가 눌리면 이륙
        drone.takeoff()
    if keyboard.is_pressed("DOWN"):  # 'DOWN' 키가 눌리면 착륙
        drone.land()

    return [left_right, front_back, up_down, clock_counter]

# 1️⃣ 드론 객체 생성
drone = Tello()

# 2️⃣ 드론과 연결
drone.connect()

# 3️⃣ 배터리 상태 출력
print(drone.get_battery())  # 배터리 퍼센트 출력

# 비디오 스트리밍을 별도의 스레드에서 실행
drone.streamon()

# 비디오 스트리밍 스레드 시작
video_thread = threading.Thread(target=video_stream_thread)
video_thread.daemon = True  # 메인 프로그램 종료 시 스레드도 종료됨
video_thread.start()

while True:
    results = getInput()
    drone.send_rc_control(results[0], results[1], results[2], results[3])
    sleep(1)

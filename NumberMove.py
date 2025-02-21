from time import sleep
from djitellopy import Tello
import cv2
import math

# 드론 제어 함수
def move_drone(x, y, z, time=1):
    drone.send_rc_control(x, y, z, 0)
    sleep(time)

# 1 만들기 (앞으로 조금 가기)
def move_one():
    move_drone(0, 20, 0, 1)  # 앞쪽으로 조금 이동

# 2 만들기 (오른쪽으로 둥글게 왼쪽으로 내려오고 다시 오른쪽으로 가기)
def move_two():
    # 오른쪽으로 이동
    move_drone(20, 0, 0, 1)
    # 왼쪽으로 둥글게 내려오며
    for i in range(30):
        move_drone(0, -2, 0, 0.05)  # 왼쪽으로 둥글게 내려감
    # 오른쪽으로 다시 이동
    move_drone(20, 0, 0, 1)

# 3 만들기 (앞으로 조금 가고, 오른쪽으로 돌아가며 내려오기)
def move_three():
    move_drone(0, 20, 0, 1)  # 앞쪽으로 조금 이동
    for i in range(30):
        move_drone(2, -2, 0, 0.05)  # 오른쪽으로 돌아가며 내려옴

# 4 만들기 (왼쪽으로 올라가면서 오른쪽으로 가기)
def move_four():
    move_drone(-10, 10, 0, 1)  # 왼쪽으로 올라가며
    move_drone(10, 0, 0, 1)    # 오른쪽으로 이동

# 5 만들기 (오른쪽으로 이동 후 아래로 내려가기)
def move_five():
    move_drone(20, 0, 0, 1)  # 오른쪽으로 이동
    move_drone(0, -20, 0, 1)  # 아래로 내려가기

# 6 만들기 (왼쪽으로 돌아가며 내려오기)
def move_six():
    for i in range(30):
        move_drone(-2, -2, 0, 0.05)  # 왼쪽으로 돌아가며 내려감

# 7 만들기 (오른쪽으로 이동 후 돌기)
def move_seven():
    move_drone(20, 0, 0, 1)  # 오른쪽으로 이동
    for i in range(30):
        move_drone(0, 2, 0, 0.05)  # 오른쪽으로 돌며 올라감

# 8 만들기 (오른쪽으로 직선, 왼쪽으로 돌고 내려오기)
def move_eight():
    move_drone(20, 0, 0, 1)  # 오른쪽으로 직선 이동
    for i in range(30):
        move_drone(-2, 2, 0, 0.05)  # 왼쪽으로 돌며 내려옴
    move_drone(20, 0, 0, 1)  # 오른쪽으로 다시 이동

# 9 만들기 (오른쪽으로 직선, 왼쪽으로 올라가며 내려오기)
def move_nine():
    move_drone(20, 0, 0, 1)  # 오른쪽으로 직선 이동
    for i in range(30):
        move_drone(2, 2, 0, 0.05)  # 왼쪽으로 올라가며 이동
    move_drone(0, -20, 0, 1)  # 아래로 내려가기

# 123456789 모양 만들기
def create_numbers():
    move_one()  # 1
    move_two()  # 2
    move_three()  # 3
    move_four()  # 4
    move_five()  # 5
    move_six()  # 6
    move_seven()  # 7
    move_eight()  # 8
    move_nine()  # 9

# 비디오 스트림 함수
def video_stream():
    image = drone.get_frame_read().frame
    image = cv2.resize(image, (500, 460))
    cv2.imshow("Image", image)
    cv2.waitKey(1)

# 드론 객체 생성
drone = Tello()

# 드론과 연결
drone.connect()

# 배터리 상태 출력
print(drone.get_battery())  # 배터리 퍼센트 출력

# 드론 스트리밍 시작
drone.streamon()

# 이륙
drone.takeoff()

# 123456789 모양 만들기
create_numbers()

# 착륙
drone.land()

cv2.destroyAllWindows()  # 영상 창 닫기

from djitellopy import Tello
import cv2

# 드론 초기화 및 연결
drone = Tello()
drone.connect()
print(f"Battery: {drone.get_battery()}%")

# 스트리밍 시작
drone.streamon()

while True:
    frame = drone.get_frame_read().frame
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Tello Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 정리
drone.streamoff()
cv2.destroyAllWindows()

from djitellopy import Tello
import cv2
import time

# 1. 드론 초기화
tello = Tello()
tello.connect()
tello.streamon()

# 이륙
tello.takeoff()

try:
    while True:
        # 2. 센서 데이터 수집 (카메라 프레임)
        frame = tello.get_frame_read().frame
        frame = cv2.resize(frame, (640, 480))
        
        # 3. 장애물 감지 (단순히 화면 중앙의 픽셀 밝기 기반)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        center_brightness = gray[240, 320]

        if center_brightness < 50:  # 어두운 물체가 가까이 있는 경우
            print("⚠ 장애물 감지됨! 회피 동작 실행")
            
            # 4. 회피 동작 (예시: 왼쪽으로 회피)
            tello.move_left(30)
            time.sleep(1)
            
            # 5. 경로 복귀
            tello.move_right(30)
            time.sleep(1)
        else:
            print("🟢 경로 정상")
        
        # 영상 출력
        cv2.imshow("Tello Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # 종료 처리
    tello.land()
    tello.streamoff()
    cv2.destroyAllWindows()

from djitellopy import tello
import cv2
import os
from datetime import datetime

# 색감 보정 함수
def adjust_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR → RGB 변환
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=20)  # 밝기 & 대비 조정
    return img

# 드론 연결
drone = tello.Tello()
drone.connect()
print(f"Battery: {drone.get_battery()}%")

# 드론 비디오 스트리밍 시작
drone.streamon()

# 사진을 저장할 폴더 지정
folder_path = "C:\\captured_images"  # 저장할 폴더 경로 (Windows에서는 \\ 사용)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 실시간 영상 및 사진 촬영
while True:
    img = drone.get_frame_read().frame
    img = cv2.resize(img, (640, 480))  # 크기 조정

    # 이미지 색감 보정 및 화면 좌우 반전
    img = cv2.flip(img, 1)  # 화면 좌우 반전
    img = adjust_image(img)  # 색감 보정

    # 화면 표시
    cv2.imshow('frame', img)

    # 's' 키를 눌러 사진을 찍고 저장
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # 's' 키 눌렀을 때
        # 파일 이름에 현재 시간을 추가하여 고유하게 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_name = f"image_{timestamp}.jpg"
        img_path = os.path.join(folder_path, img_name)
        
        # 사진 저장
        cv2.imwrite(img_path, img)
        print(f"Image saved: {img_path}")

    # 'q' 키를 눌러 종료
    elif key == ord('q'):
        break

# 드론 스트리밍 종료 및 카메라 창 닫기
drone.streamoff()
cv2.destroyAllWindows()

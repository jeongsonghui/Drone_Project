from djitellopy import Tello
import cv2
import os

tello = Tello()
tello.connect()
tello.streamon()

frame_read = tello.get_frame_read()
i = 0

save_path = r"C:\Users\Administrator\Desktop\Lifter"


while True:
    frame = frame_read.frame
    cv2.imshow("Drone View", frame)

    key = cv2.waitKey(1)
    
    # S 누르면 프레임 저장
    if key == ord('s'):
        filename = os.path.join(save_path, f"lifter_{i}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        i += 1

    # Q 누르면 종료
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
tello.end()

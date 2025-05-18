import os

folder_path = "C:\\captured_images"  # 이미지 폴더 경로

# 이미지 확장자 목록
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

# 폴더 내 이미지 파일 수 세기
image_count = len([file for file in os.listdir(folder_path) if file.lower().endswith(image_extensions)])

print(f"이미지 개수: {image_count}")

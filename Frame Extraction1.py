import cv2
import os

# 影片檔案路徑
video_path = "videos/dance.mp4"

# 拆出的圖片存放資料夾
output_dir = "dataset/images"
os.makedirs(output_dir, exist_ok=True)

# 開啟影片
cap = cv2.VideoCapture(video_path)
frame_count = 0
save_every_n_frame = 5  # 每 5 幀存一張圖片（可依需求調整）

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % save_every_n_frame == 0:
        img_name = f"frame_{frame_count:05d}.jpg"
        cv2.imwrite(os.path.join(output_dir, img_name), frame)

    frame_count += 1

cap.release()
print(f"已拆出 {frame_count} 幀圖片到 {output_dir}")
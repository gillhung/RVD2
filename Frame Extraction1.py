import cv2
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilenames

# 隱藏 tkinter 主視窗
Tk().withdraw()

# 多選影片
video_paths = askopenfilenames(
    title="請選擇影片",
    filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
)

# 如果沒有選影片
if not video_paths:
    print("未選擇影片")
    exit()

# 每幾幀存一張
save_every_n_frame = 5

# 處理每部影片
for video_path in video_paths:

    # 取得影片名稱（不含副檔名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # 建立該影片專屬資料夾
    output_dir = os.path.join("dataset/images", video_name)
    os.makedirs(output_dir, exist_ok=True)

    # 開啟影片
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 每 N 幀存一次
        if frame_count % save_every_n_frame == 0:

            img_name = f"{video_name}_{saved_count:05d}.jpg"

            cv2.imwrite(
                os.path.join(output_dir, img_name),
                frame
            )

            saved_count += 1

        frame_count += 1

    cap.release()

    print(f"{video_name} 拆出 {saved_count} 張圖片")

print("全部影片處理完成")
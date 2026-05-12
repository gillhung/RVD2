import os
import json
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 設定路徑 
IMAGE_ROOT = "dataset/single_person_images"
OUT_ROOT = "annotations/mediapipe_pose"
MODEL_PATH = "pose_landmarker_full.task"

os.makedirs(OUT_ROOT, exist_ok=True)

# 載入 Pose Landmarker 模型 
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)
landmarker = vision.PoseLandmarker.create_from_options(options)

# 找出所有子資料夾中的圖片
image_tasks = []
for root, dirs, files in os.walk(IMAGE_ROOT):
    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            # 圖片完整路徑
            img_path = os.path.join(root, file)
            # 取得相對路徑（為了保持資料夾結構）
            rel_path = os.path.relpath(root, IMAGE_ROOT)
            # 設定輸出 JSON 的資料夾
            out_dir = os.path.join(OUT_ROOT, rel_path)
            
            image_tasks.append({
                "img_path": img_path,
                "out_dir": out_dir,
                "file_name": file
            })

print(f"📊 找到 {len(image_tasks)} 張裁切後的個人圖片，開始偵測關鍵點...")

# 對每張圖跑 Pose 偵測
success_count = 0

for t in image_tasks:
    # 確保輸出子資料夾存在
    os.makedirs(t["out_dir"], exist_ok=True)

    # 讀取圖片（取得寬高以便還原座標）
    image_cv = cv2.imread(t["img_path"])
    if image_cv is None:
        continue
    h, w, _ = image_cv.shape

    # MediaPipe 偵測
    try:
        mp_image = mp.Image.create_from_file(t["img_path"])
        result = landmarker.detect(mp_image)
    except Exception as e:
        print(f"❌ 偵測失敗: {t['file_name']} | {e}")
        continue

    if not result.pose_landmarks:
        # print(f"⚠ 未偵測到人體: {t['file_name']}")
        continue

    # 整理關鍵點座標
    # MediaPipe 回傳的是 0~1 的比例，需要乘上寬高
    keypoints = []
    for lm in result.pose_landmarks[0]:
        x = lm.x * w
        y = lm.y * h
        score = lm.visibility if hasattr(lm, 'visibility') else 0
        keypoints.append([round(x, 2), round(y, 2), round(score, 4)])

    # 儲存 JSON
    base_name = os.path.splitext(t["file_name"])[0]
    out_path = os.path.join(t["out_dir"], f"{base_name}.json")
    
    with open(out_path, "w") as f:
        json.dump(keypoints, f, indent=4)

    success_count += 1
    if success_count % 10 == 0:
        print(f"✅ 已處理 {success_count} 張圖片...")

print(f"\n===== 🎉 全部完成 =====")
print(f"成功偵測關鍵點並存檔: {success_count} 筆")
print(f"輸出路徑: {OUT_ROOT}")
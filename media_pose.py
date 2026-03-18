import os
import json
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

IMAGE_DIR = "dataset/single_person_images"
OUT_DIR = "annotations/mediapipe_pose"
os.makedirs(OUT_DIR, exist_ok=True)

# 載入 Pose Landmarker 模型
base_options = python.BaseOptions(
    model_asset_path="pose_landmarker_full.task"
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)

landmarker = vision.PoseLandmarker.create_from_options(options)

# 對每張圖跑 Pose
for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    image = cv2.imread(img_path)
    h, w, _ = image.shape

    # 正確的新 API
    mp_image = mp.Image.create_from_file(img_path)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        continue

    keypoints = []
    for lm in result.pose_landmarks[0]:
        x = lm.x * w
        y = lm.y * h
        score = lm.visibility
        keypoints.append([x, y, score])

    with open(
        os.path.join(OUT_DIR, img_name.replace(".jpg", ".json")),
        "w"
    ) as f:
        json.dump(keypoints, f)

    print(f"{img_name}: {len(keypoints)} keypoints")

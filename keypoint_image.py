import os
import cv2
import json

# 路徑設定
IMAGE_DIR = "dataset/single_person_images"
COCO_JSON_PATH = "annotations/coco_annotations.json"
OUT_DIR = "annotations/marked_images"
os.makedirs(OUT_DIR, exist_ok=True)

# 讀取 COCO JSON
with open(COCO_JSON_PATH, 'r') as f:
    coco = json.load(f)

# image_id -> file_name
img_dict = {img['id']: img['file_name'] for img in coco['images']}

# 調整比例
bbox_margin_ratio = 0.15  # bbox 上下左右各加 15%
keypoint_radius = 8       # 關鍵點半徑

# 畫圖
for ann in coco['annotations']:
    img_name = img_dict[ann['image_id']]
    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print("⚠️ 找不到圖片:", img_name)
        continue

    h_img, w_img = img.shape[:2]

    # 原 bbox
    x, y, w, h = map(int, ann['bbox'])

    # 擴大 bbox
    x_margin = int(w * bbox_margin_ratio)
    y_margin = int(h * bbox_margin_ratio)
    x1 = max(0, x - x_margin)
    y1 = max(0, y - y_margin)
    x2 = min(w_img - 1, x + w + x_margin)
    y2 = min(h_img - 1, y + h + y_margin)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 畫 keypoints
    kpts = ann['keypoints']
    for i in range(0, len(kpts), 3):
        xk, yk, v = kpts[i], kpts[i+1], kpts[i+2]
        if v > 0:
            cv2.circle(img, (int(xk), int(yk)), keypoint_radius, (0, 0, 255), -1)

    # 存檔
    out_path = os.path.join(OUT_DIR, img_name)
    cv2.imwrite(out_path, img)

print(f"✅ 標記完成，圖片存放於: {OUT_DIR}")
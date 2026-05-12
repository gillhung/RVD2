import os
import json
from ultralytics import YOLO

# --- 設定區 ---
MODEL_PATH = "yolov8n-pose.pt"
IMAGE_DIR = "dataset/images"
OUT_ROOT = "annotations/yolo_boxes"
CONF_TH = 0.3
BBOX_EXPAND = 1.25

# 初始化 
model = YOLO(MODEL_PATH)

def expand_bbox(bbox, scale=1.25):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    return [
        max(cx - w / 2, 0),
        max(cy - h / 2, 0),
        cx + w / 2,
        cy + h / 2
    ]

# 1. 找出所有圖片（包含子資料夾）
image_tasks = []
for root, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 取得圖片完整路徑
            img_path = os.path.join(root, file)
            # 取得相對於 IMAGE_DIR 的路徑，用來在輸出目錄建立對應結構
            rel_path = os.path.relpath(root, IMAGE_DIR)
            out_dir = os.path.join(OUT_ROOT, rel_path)
            
            image_tasks.append((img_path, out_dir, file))

print(f"🚀 找到 {len(image_tasks)} 張圖片，開始處理...")

# 2. 執行偵測
for img_path, out_dir, img_name in image_tasks:
    # 自動建立對應的輸出資料夾
    os.makedirs(out_dir, exist_ok=True)
    
    # 執行推理
    results = model.predict(source=img_path, save=False, verbose=False)
    
    bboxes = []
    for r in results:
        if r.boxes is None:
            continue

        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            if int(cls) != 0:  # 只抓 person 類別
                continue
            if conf < CONF_TH:
                continue

            # 轉為 list 並擴展邊框
            bbox = box.tolist()
            bbox = expand_bbox(bbox, BBOX_EXPAND)
            bboxes.append(bbox)

    # 儲存 JSON (檔名維持一致，只改副檔名)
    base_name = os.path.splitext(img_name)[0]
    out_path = os.path.join(out_dir, f"{base_name}.json")
    
    with open(out_path, "w") as f:
        json.dump(bboxes, f, indent=4)

    print(f"✅ 已處理: {img_name} ({len(bboxes)} 人)")

print("\n✨ 全部處理完成！JSON 檔案儲存在:", OUT_ROOT)
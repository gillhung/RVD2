import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import json
from collections import defaultdict
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

# 1. 找出所有圖片，按資料夾分組（確保追蹤順序正確）
folder_tasks = defaultdict(list)
for root, dirs, files in os.walk(IMAGE_DIR):
    for file in sorted(files):  # 排序確保幀順序正確
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(root, file)
            rel_path = os.path.relpath(root, IMAGE_DIR)
            out_dir = os.path.join(OUT_ROOT, rel_path)
            folder_tasks[out_dir].append((img_path, file))

total_images = sum(len(v) for v in folder_tasks.values())
print(f"🚀 找到 {total_images} 張圖片，共 {len(folder_tasks)} 個資料夾，開始處理...")

total_suspicious = 0

# 2. 每個資料夾獨立處理（同一段影片的幀放一起追蹤）
for out_dir, img_list in sorted(folder_tasks.items()):
    os.makedirs(out_dir, exist_ok=True)

    prev_ids = set()
    suspicious = []

    for img_path, img_name in img_list:

        # 用 track 取代 predict，保持跨幀 ID 一致，解決 ID Switch
        results = model.track(
            source=img_path,
            persist=True,   # 跨幀保持追蹤狀態
            conf=CONF_TH,
            verbose=False
        )

        bboxes = []
        curr_ids = set()

        for r in results:
            if r.boxes is None:
                continue

            boxes = r.boxes
            for i, (box, cls, conf) in enumerate(zip(
                boxes.xyxy, boxes.cls, boxes.conf
            )):
                if int(cls) != 0:  # 只抓 person 類別
                    continue
                if float(conf) < CONF_TH:
                    continue

                # 取得追蹤 ID
                track_id = int(boxes.id[i]) if boxes.id is not None else -1
                curr_ids.add(track_id)

                bbox = expand_bbox(box.tolist(), BBOX_EXPAND)
                bboxes.append({
                    'bbox': bbox,
                    'track_id': track_id,
                    'conf': round(float(conf), 4)
                })

        # ID Switch 檢查
        if prev_ids and curr_ids:
            disappeared = prev_ids - curr_ids
            appeared = curr_ids - prev_ids
            if disappeared and appeared:
                suspicious.append({
                    'frame': img_name,
                    'reason': f'ID 消失:{list(disappeared)}，新出現:{list(appeared)}，可能發生 ID Switch'
                })

        prev_ids = curr_ids

        # 儲存 bbox JSON
        base_name = os.path.splitext(img_name)[0]
        out_path = os.path.join(out_dir, f"{base_name}.json")
        with open(out_path, "w") as f:
            json.dump(bboxes, f, indent=4)

    # 存可疑幀清單
    if suspicious:
        suspicious_path = os.path.join(out_dir, "suspicious.json")
        with open(suspicious_path, "w", encoding="utf-8") as f:
            json.dump(suspicious, f, indent=2, ensure_ascii=False)
        total_suspicious += len(suspicious)
        print(f"  ⚠️  {out_dir}：{len(suspicious)} 個可疑幀")

print(f"\n✨ 全部處理完成！JSON 檔案儲存在：{OUT_ROOT}")
print(f"⚠️  總可疑幀：{total_suspicious} 個，請人工確認各資料夾內的 suspicious.json")
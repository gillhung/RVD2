import os
import json
import cv2

# 路徑設定
MP_JSON_ROOT = "annotations/mediapipe_pose"
COCO_JSON_PATH = "annotations/coco_annotations.json"
IMAGE_ROOT = "dataset/single_person_images"

# MediaPipe (33點) → COCO (17點) 的對應 index
mp_to_coco_idx = [
    0, 1, 4, 3, 7,
    11, 12, 13, 14,
    15, 16, 23, 24,
    25, 26, 27, 28
]

coco_kpts_name = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

skeleton = [
    [16,14],[14,12],[17,15],[15,13],
    [12,13],[6,12],[7,13],[6,7],
    [6,8],[7,9],[8,10],[9,11],
    [2,3],[1,2],[1,3],[2,4],[3,5]
]

coco_dict = {
    "images": [],
    "annotations": [],
    "categories": [{
        "id": 1,
        "name": "person",
        "supercategory": "person",
        "keypoints": coco_kpts_name,
        "skeleton": skeleton
    }]
}

ann_id = 1
img_id = 1
valid_count = 0

# 使用 os.walk 遍歷所有子資料夾
json_tasks = []
for root, dirs, files in os.walk(MP_JSON_ROOT):
    for file in files:
        if file.endswith(".json"):
            json_path = os.path.join(root, file)
            # 取得相對於 MP_JSON_ROOT 的路徑，以便去對應 IMAGE_ROOT 裡的圖片
            rel_path = os.path.relpath(root, MP_JSON_ROOT)
            
            # 對應圖片的路徑 (維持同樣的子資料夾結構)
            img_name = file.replace(".json", ".jpg")
            img_path = os.path.join(IMAGE_ROOT, rel_path, img_name)
            
            json_tasks.append({
                "json_path": json_path,
                "img_path": img_path,
                "file_name": file,
                "rel_dir": rel_path # 存下來備用，存進 COCO 時 file_name 要包含路徑
            })

print(f" 找到 {len(json_tasks)} 個關鍵點檔案，開始轉換為 COCO 格式...")

# 主迴圈處理任務
for t in sorted(json_tasks, key=lambda x: x['file_name']):
    if not os.path.exists(t["img_path"]):
        # print(f"❌ 找不到圖片: {t['img_path']}")
        continue

    img = cv2.imread(t["img_path"])
    if img is None:
        continue

    h, w = img.shape[:2]

    with open(t["json_path"], "r") as f:
        mp_kpts = json.load(f)

    coco_kpts = []
    xs, ys = [], []
    num_keypoints = 0

    # 轉換關鍵點
    for idx in mp_to_coco_idx:
        if idx < len(mp_kpts):
            x, y, score = mp_kpts[idx]
            if score > 0:
                v = 2
                num_keypoints += 1
                xs.append(x)
                ys.append(y)
            else:
                v = 0
            coco_kpts.extend([float(x), float(y), v])
        else:
            coco_kpts.extend([0.0, 0.0, 0])

    # 過濾爛資料 (至少要有 5 個點)
    if num_keypoints < 5:
        continue

    # 計算 bbox
    x_min = max(0, min(xs))
    y_min = max(0, min(ys))
    x_max = min(w, max(xs))
    y_max = min(h, max(ys))
    bw, bh = x_max - x_min, y_max - y_min

    if bw <= 10 or bh <= 10:
        continue

    bbox = [float(x_min), float(y_min), float(bw), float(bh)]
    area = float(bw * bh)

    # 寫入 COCO 結構
    # 注意：file_name 建議保留相對路徑，訓練時才找得到
    file_rel_name = os.path.join(t["rel_dir"], os.path.basename(t["img_path"]))
    
    coco_dict["images"].append({
        "file_name": file_rel_name.replace("\\", "/"), # 統一用斜線避免 Windows 問題
        "height": h,
        "width": w,
        "id": img_id
    })

    coco_dict["annotations"].append({
        "id": ann_id,
        "image_id": img_id,
        "category_id": 1,
        "keypoints": coco_kpts,
        "num_keypoints": num_keypoints,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
        "bbox_score": 1.0 
    })

    ann_id += 1
    img_id += 1
    valid_count += 1

# 存檔
os.makedirs(os.path.dirname(COCO_JSON_PATH), exist_ok=True)
with open(COCO_JSON_PATH, "w") as f:
    json.dump(coco_dict, f, indent=4)

print(f"\n✅ 全部轉換完成！")
print(f"有效標記數量: {valid_count}")
print(f"結果檔案: {COCO_JSON_PATH}")
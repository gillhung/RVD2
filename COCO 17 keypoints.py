import os
import json
import cv2

# 路徑設定
MP_JSON_DIR = "annotations/mediapipe_pose"
COCO_JSON_PATH = "annotations/coco_annotations.json"
IMAGE_DIR = "dataset/single_person_images"

# MediaPipe → COCO mapping
mp_to_coco_idx = [
    0, 1, 4, 3, 7,
    11, 12, 13, 14,
    15, 16, 23, 24,
    25, 26, 27, 28
]

# ===== COCO keypoints =====
coco_kpts_name = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# ===== skeleton =====
skeleton = [
    [16,14],[14,12],[17,15],[15,13],
    [12,13],[6,12],[7,13],[6,7],
    [6,8],[7,9],[8,10],[9,11],
    [2,3],[1,2],[1,3],[2,4],[3,5]
]

# ===== COCO dict =====
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

# ===== 主迴圈 =====
for fname in sorted(os.listdir(MP_JSON_DIR)):
    if not fname.endswith(".json"):
        continue

    img_path = os.path.join(IMAGE_DIR, fname.replace(".json", ".jpg"))
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    with open(os.path.join(MP_JSON_DIR, fname), "r") as f:
        mp_kpts = json.load(f)

    coco_kpts = []
    xs, ys = [], []
    num_keypoints = 0

    # ===== keypoints =====
    for idx in mp_to_coco_idx:
        if idx < len(mp_kpts):
            x, y, score = mp_kpts[idx]

            # 🔥 關鍵：放寬條件（避免被全部丟掉）
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

    # ===== 過濾爛資料 =====
    if num_keypoints < 5:
        continue

    # ===== bbox =====
    x_min = max(0, min(xs))
    y_min = max(0, min(ys))
    x_max = min(w, max(xs))
    y_max = min(h, max(ys))

    bw = x_max - x_min
    bh = y_max - y_min

    if bw <= 10 or bh <= 10:
        continue

    bbox = [float(x_min), float(y_min), float(bw), float(bh)]
    area = float(bw * bh)

    # ===== images =====
    coco_dict["images"].append({
        "file_name": os.path.basename(img_path),
        "height": h,
        "width": w,
        "id": img_id
    })

    # ===== annotations =====
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

# ===== 存檔 =====
os.makedirs(os.path.dirname(COCO_JSON_PATH), exist_ok=True)
with open(COCO_JSON_PATH, "w") as f:
    json.dump(coco_dict, f, indent=4)

print(f"✅ 完成！有效資料數量: {valid_count}")
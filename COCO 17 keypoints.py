import os
import json

# 路徑設定
MP_JSON_DIR = "annotations/mediapipe_pose"  # 33點 JSON
COCO_JSON_PATH = "annotations/coco_annotations.json"
IMAGE_DIR = "dataset/single_person_images"

# COCO keypoints 對應 MediaPipe 33 點索引
mp_to_coco_idx = [
    0,  # 鼻子
    1,  # 左眼
    4,  # 右眼
    3,  # 左耳
    7,  # 右耳
    11, # 左肩
    12, # 右肩
    13, # 左肘
    14, # 右肘
    15, # 左手腕
    16, # 右手腕
    23, # 左臀
    24, # 右臀
    25, # 左膝
    26, # 右膝
    27, # 左踝
    28  # 右踝
]

# COCO keypoints 名稱
coco_kpts_name = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# COCO skeleton
skeleton = [
    [16, 14], [14,12], [17,15], [15,13],
    [12,13],[6,12],[7,13],[6,7],
    [6,8],[7,9],[8,10],[9,11],
    [2,3],[1,2],[1,3],[2,4],[3,5]
]

# 建立 COCO 結構
coco_dict = {
    "images": [],
    "annotations": [],
    "categories": [
        {
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": coco_kpts_name,
            "skeleton": skeleton
        }
    ]
}

ann_id = 1
img_id = 1

for fname in sorted(os.listdir(MP_JSON_DIR)):
    if not fname.lower().endswith(".json"):
        continue

    img_path = os.path.join(IMAGE_DIR, fname.replace(".json",".jpg"))
    if not os.path.exists(img_path):
        continue

    # 讀取圖片大小
    import cv2
    img_cv = cv2.imread(img_path)
    h, w, _ = img_cv.shape

    # images
    coco_dict["images"].append({
        "file_name": os.path.basename(img_path),
        "height": h,
        "width": w,
        "id": img_id
    })

    # 讀取 MediaPipe JSON
    with open(os.path.join(MP_JSON_DIR, fname), "r") as f:
        mp_kpts = json.load(f)

    # 轉成 COCO 17 keypoints
    coco_kpts = []
    xs, ys = [], []
    for idx in mp_to_coco_idx:
        if idx < len(mp_kpts):
            x, y, score = mp_kpts[idx]
            coco_kpts.extend([x, y, score])
            xs.append(x)
            ys.append(y)
        else:
            # 如果缺點，填 0
            coco_kpts.extend([0,0,0])

    # 計算 bbox
    if xs and ys:
        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
    else:
        bbox = [0,0,0,0]

    # annotations
    coco_dict["annotations"].append({
        "id": ann_id,
        "image_id": img_id,
        "category_id": 1,
        "keypoints": coco_kpts,
        "num_keypoints": sum([1 for s in coco_kpts[2::3] if s>0]),
        "bbox": bbox,
        "iscrowd": 0
    })

    ann_id += 1
    img_id += 1

# 存成 COCO JSON
os.makedirs(os.path.dirname(COCO_JSON_PATH), exist_ok=True)
with open(COCO_JSON_PATH, "w") as f:
    json.dump(coco_dict, f, indent=4)

print(f"✅ COCO JSON 已生成: {COCO_JSON_PATH}")
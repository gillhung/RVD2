import json
import random
import os

# ====== 設定 ======
INPUT_JSON = "annotations/coco_annotations.json"
OUTPUT_DIR = "annotations"
SPLIT_RATIO = 0.8   # train 比例

# ====== 讀取 JSON ======
with open(INPUT_JSON, "r") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

# ====== 打亂圖片 ======
random.shuffle(images)

# ====== 切分 ======
split_idx = int(len(images) * SPLIT_RATIO)
train_images = images[:split_idx]
val_images = images[split_idx:]

# ====== 建立 image_id 對應 ======
train_img_ids = set(img["id"] for img in train_images)
val_img_ids = set(img["id"] for img in val_images)

# ====== 分 annotations ======
train_annotations = []
val_annotations = []

for ann in annotations:
    if ann["image_id"] in train_img_ids:
        train_annotations.append(ann)
    elif ann["image_id"] in val_img_ids:
        val_annotations.append(ann)

# ====== 建立 COCO 結構 ======
train_coco = {
    "images": train_images,
    "annotations": train_annotations,
    "categories": categories
}

val_coco = {
    "images": val_images,
    "annotations": val_annotations,
    "categories": categories
}

# ====== 存檔 ======
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_path = os.path.join(OUTPUT_DIR, "train.json")
val_path = os.path.join(OUTPUT_DIR, "val.json")

with open(train_path, "w") as f:
    json.dump(train_coco, f, indent=4)

with open(val_path, "w") as f:
    json.dump(val_coco, f, indent=4)

print("✅ 完成資料切分！")
print(f"Train images: {len(train_images)}")
print(f"Val images: {len(val_images)}")
import os
import json

IMAGE_DIR = "dataset/single_person_images"
COCO_JSON_PATH = "annotations/coco_annotations.json"

with open(COCO_JSON_PATH, 'r') as f:
    coco = json.load(f)

missing = []
for img in coco['images']:
    path = os.path.join(IMAGE_DIR, img['file_name'])
    if not os.path.exists(path):
        missing.append(img['file_name'])

if missing:
    print("⚠️ 這些圖片在資料夾找不到:", missing)
else:
    print("✅ 所有 COCO JSON 中的圖片都存在。")
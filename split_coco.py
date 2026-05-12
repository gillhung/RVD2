import json
import random
import os

# 設定
INPUT_JSON = "annotations/coco_annotations.json"
OUTPUT_DIR = "annotations"
SPLIT_RATIO = 0.8  # 80% 訓練, 20% 驗證
RANDOM_SEED = 42   # 設定隨機種子，確保每次執行切分的結果都一樣 (實驗可重複性)

# 檢查輸入 
if not os.path.exists(INPUT_JSON):
    print(f"❌ 找不到來源標記檔: {INPUT_JSON}")
    exit()

# 讀取 JSON 
with open(INPUT_JSON, "r") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

print(f"📊 總計圖片數: {len(images)}")
print(f"📊 總計標記數: {len(annotations)}")

# 打亂圖片
# 使用固定種子，這樣如果你之後增加資料再跑一次，原本的 train/val 歸類比較不會大洗牌
random.seed(RANDOM_SEED)
random.shuffle(images)

# 切分 
split_idx = int(len(images) * SPLIT_RATIO)
train_images = images[:split_idx]
val_images = images[split_idx:]

# 建立 image_id 對應 (使用 set 加快搜尋速度) 
train_img_ids = set(img["id"] for img in train_images)
val_img_ids = set(img["id"] for img in val_images)

# 分配 annotations
train_annotations = [ann for ann in annotations if ann["image_id"] in train_img_ids]
val_annotations = [ann for ann in annotations if ann["image_id"] in val_img_ids]

# 建立新 COCO 結構 
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

# 存檔 
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_path = os.path.join(OUTPUT_DIR, "train.json")
val_path = os.path.join(OUTPUT_DIR, "val.json")

# 存檔 train.json
with open(train_path, "w") as f:
    json.dump(train_coco, f, indent=4)

# 存檔 val.json
with open(val_path, "w") as f:
    json.dump(val_coco, f, indent=4)

print("\n" + "="*30)
print("✅ 完成資料切分！")
print(f"📂 訓練集 (Train): {len(train_images)} 張圖片, {len(train_annotations)} 個標記")
print(f"📂 驗證集 (Val):   {len(val_images)} 張圖片, {len(val_annotations)} 個標記")
print(f"📝 存檔位置: {OUTPUT_DIR}")
print("="*30)
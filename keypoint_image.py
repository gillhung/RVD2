import os
import cv2
import json

# 路徑設定
IMAGE_DIR = "dataset/single_person_images"
COCO_JSON_PATH = "annotations/coco_annotations.json"
OUT_DIR = "annotations/marked_images"
os.makedirs(OUT_DIR, exist_ok=True)

# 讀取 COCO JSON
if not os.path.exists(COCO_JSON_PATH):
    print(f"❌ 找不到 JSON 檔案: {COCO_JSON_PATH}")
    exit()

with open(COCO_JSON_PATH, 'r') as f:
    coco = json.load(f)

# 建立映射表：image_id -> file_name
img_dict = {img['id']: img['file_name'] for img in coco['images']}

# 繪圖參數調整
bbox_margin_ratio = 0.15  # bbox 邊緣留白比例
keypoint_radius = 5       # 關鍵點半徑 (稍微縮小一點比較美觀)

print(f"🎨 開始繪製標記圖片，預計處理 {len(coco['annotations'])} 個標記...")

# 開始畫圖
success_count = 0

for ann in coco['annotations']:
    img_name = img_dict[ann['image_id']] # 這裡的 img_name 已經包含子路徑，如 "dance/001.jpg"
    img_path = os.path.join(IMAGE_DIR, img_name)
    
    # 讀取圖片
    img = cv2.imread(img_path)
    if img is None:
        # 嘗試處理 Windows/Linux 路徑斜線差異
        img = cv2.imread(img_path.replace("/", "\\"))
        if img is None:
            print(f"⚠️ 找不到圖片: {img_path}")
            continue

    h_img, w_img = img.shape[:2]

    # 畫 Bounding Box
    x, y, w, h = map(int, ann['bbox'])
    x_margin = int(w * bbox_margin_ratio)
    y_margin = int(h * bbox_margin_ratio)
    
    x1 = max(0, x - x_margin)
    y1 = max(0, y - y_margin)
    x2 = min(w_img - 1, x + w + x_margin)
    y2 = min(h_img - 1, y + h + y_margin)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 畫 Keypoints 
    kpts = ann['keypoints']
    for i in range(0, len(kpts), 3):
        xk, yk, v = kpts[i], kpts[i+1], kpts[i+2]
        if v > 0:
            # 畫圓圈表示關鍵點
            cv2.circle(img, (int(xk), int(yk)), keypoint_radius, (0, 0, 255), -1)

    # 存檔處理 (支援子資料夾)
    out_path = os.path.join(OUT_DIR, img_name)
    out_subdir = os.path.dirname(out_path)
    
    # ⭐ 關鍵：如果子資料夾不存在，必須先建立，否則 imwrite 會失敗
    os.makedirs(out_subdir, exist_ok=True)
    
    # 存檔 (考慮到可能有中文路徑，使用 imencode 比較保險)
    result, encoded = cv2.imencode(".jpg", img)
    if result:
        encoded.tofile(out_path)
        success_count += 1
    else:
        print(f"❌ 存檔失敗: {out_path}")

print(f"\n✅ 標記完成！")
print(f"成功產出圖片數: {success_count}")
print(f"請到此查看成果: {os.path.abspath(OUT_DIR)}")
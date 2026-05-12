import os
import cv2
import json

# 1️⃣ 基本路徑設定
current_dir = os.path.dirname(os.path.abspath(__file__))
json_root = os.path.join(current_dir, "annotations", "yolo_boxes")
image_root = os.path.join(current_dir, "dataset", "images")
output_root = os.path.join(current_dir, "dataset", "single_person_images")

os.makedirs(output_root, exist_ok=True)

# 2️⃣ 找出所有 JSON 任務
tasks = []
for root, dirs, files in os.walk(json_root):
    for file in files:
        if file.endswith(".json"):
            # JSON 完整路徑
            json_path = os.path.join(root, file)
            # 取得相對路徑（例如 "dance/image1.json" 中的 "dance"）
            rel_path = os.path.relpath(root, json_root)
            
            # 對應圖片的路徑
            img_name = os.path.splitext(file)[0] + ".jpg"
            img_path = os.path.join(image_root, rel_path, img_name)
            
            # 輸出的資料夾結構
            out_dir = os.path.join(output_root, rel_path)
            
            tasks.append({
                "json_path": json_path,
                "img_path": img_path,
                "out_dir": out_dir,
                "base_name": os.path.splitext(file)[0]
            })

print(f"📊 找到 {len(tasks)} 個 JSON 任務")

# 3️⃣ 開始處理
success_count = 0

for t in tasks:
    # 讀取 JSON
    try:
        with open(t["json_path"], 'r') as f:
            bboxes = json.load(f)
    except Exception as e:
        print(f"❌ JSON 讀取失敗: {t['json_path']} | {e}")
        continue

    # 讀取圖片
    img = cv2.imread(t["img_path"])
    if img is None:
        # 有時候可能是 .png，嘗試換個副檔名看看
        t["img_path"] = t["img_path"].replace(".jpg", ".png")
        img = cv2.imread(t["img_path"])
        if img is None:
            print(f"❌ 讀不到圖片: {t['img_path']}")
            continue

    if not isinstance(bboxes, list) or len(bboxes) == 0:
        continue

    # 面積排序 (大的優先)
    bboxes = sorted(bboxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)

    # 建立對應的輸出資料夾
    os.makedirs(t["out_dir"], exist_ok=True)

    h, w = img.shape[:2]

    # 4️⃣ 裁切每個人
    for i, box in enumerate(bboxes):
        try:
            x1, y1, x2, y2 = map(int, box)
            # 防止超出邊界
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            
            person_img = img[y1:y2, x1:x2]
            if person_img.size == 0: continue

            # 命名與路徑
            save_name = f"{t['base_name']}_person{i+1}.jpg"
            save_path = os.path.join(t["out_dir"], save_name)

            # 解決中文路徑存檔
            result, encoded = cv2.imencode(".jpg", person_img)
            if result:
                encoded.tofile(save_path)
                success_count += 1
        except:
            continue

    print(f"✅ 已處理: {t['base_name']}")

print("\n===== 🎉 任務完成 =====")
print(f"成功輸出總人數圖片: {success_count}")
print(f"結果存放在: {output_root}")
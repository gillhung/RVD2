import os
import cv2
import json

# 1️⃣ 基本路徑設定

current_dir = os.path.dirname(os.path.abspath(__file__))

json_dir = os.path.join(current_dir, "annotations", "yolo_boxes")
image_dir = os.path.join(current_dir, "dataset", "images")
output_dir = os.path.join(current_dir, "dataset", "single_person_images")

os.makedirs(output_dir, exist_ok=True)

print("📂 JSON資料夾:", json_dir)
print("📂 圖片資料夾:", image_dir)
print("📂 輸出資料夾:", output_dir)

# 2️⃣ 檢查資料夾

if not os.path.exists(json_dir):
    print("❌ 找不到 JSON 資料夾")
    exit()

if not os.path.exists(image_dir):
    print("❌ 找不到圖片資料夾")
    exit()

# 3️⃣ 讀取 JSON 檔

json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

print(f"📊 找到 {len(json_files)} 個 JSON")

if len(json_files) == 0:
    print("❌ 沒有 JSON")
    exit()

# 4️⃣ 開始處理

success_count = 0

for json_file in json_files:

    json_path = os.path.join(json_dir, json_file)

    # 讀 JSON
    try:
        with open(json_path, 'r') as f:
            bboxes = json.load(f)
    except Exception as e:
        print("❌ JSON讀取失敗:", json_file, e)
        continue

    # 對應圖片
    img_name = os.path.splitext(json_file)[0] + ".jpg"
    img_path = os.path.join(image_dir, img_name)

    img = cv2.imread(img_path)

    if img is None:
        print("❌ 讀不到圖片:", img_name)
        continue

    if not isinstance(bboxes, list) or len(bboxes) == 0:
        print("⚠ 沒有bbox:", json_file)
        continue

    # ⭐ 依面積排序（前面的人 = person1）

    try:
        bboxes = sorted(
            bboxes,
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
            reverse=True
        )
    except:
        print("❌ bbox格式錯誤:", json_file)
        continue

    # 5️⃣ 裁切每個人

    h, w = img.shape[:2]

    for i, box in enumerate(bboxes):

        try:
            x1, y1, x2, y2 = map(int, box)
        except:
            print("❌ bbox錯誤:", json_file)
            continue

        # 防止超出邊界
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        person_img = img[y1:y2, x1:x2]

        if person_img.size == 0:
            print("❌ 裁切失敗:", img_name)
            continue

        # ⭐ 命名：person1, person2...
        save_name = f"{os.path.splitext(img_name)[0]}_person{i+1}.jpg"
        save_path = os.path.join(output_dir, save_name)

        # ⭐ 解決中文路徑
        result, encoded = cv2.imencode(".jpg", person_img)
        if result:
            encoded.tofile(save_path)
            success_count += 1
        else:
            print("❌ 存檔失敗:", save_name)

# 6️⃣ 完成

print("\n===== 🎉 完成 =====")
print("成功輸出圖片數:", success_count)
print("輸出資料夾:", output_dir)
import os
import json
import cv2

# 設定區
OUT_ROOT = "annotations/yolo_boxes"
IMAGE_DIR = "dataset/images"
CONFIRMED_BAD = "confirmed_bad.txt"  # 確認有問題的幀存這裡

# 讀取已確認的壞幀（避免重複看）
if os.path.exists(CONFIRMED_BAD):
    with open(CONFIRMED_BAD, 'r') as f:
        bad_frames = set(line.strip() for line in f if line.strip())
else:
    bad_frames = set()

print("操作說明：")
print("  d = 刪除這張（標記為壞幀）")
print("  k = 保留這張")
print("  q = 離開")
print("  s = 跳過可疑幀，只看正常幀")
print("-" * 40)

# 收集所有圖片和對應的 suspicious 狀態
tasks = []
for root, dirs, files in os.walk(OUT_ROOT):
    rel_path = os.path.relpath(root, OUT_ROOT)

    # 讀取這個資料夾的 suspicious.json
    suspicious_frames = set()
    suspicious_path = os.path.join(root, 'suspicious.json')
    if os.path.exists(suspicious_path):
        with open(suspicious_path, 'r', encoding='utf-8') as f:
            suspicious = json.load(f)
        for item in suspicious:
            suspicious_frames.add(item['frame'])

    for file in sorted(files):
        if not file.endswith('.json') or file == 'suspicious.json':
            continue

        base_name = os.path.splitext(file)[0]
        img_name = base_name + '.jpg'
        img_path = os.path.join(IMAGE_DIR, rel_path, img_name)

        if not os.path.exists(img_path):
            img_name = base_name + '.png'
            img_path = os.path.join(IMAGE_DIR, rel_path, img_name)
            if not os.path.exists(img_path):
                continue

        rel_img = os.path.join(rel_path, img_name).replace("\\", "/")

        # 跳過已確認的壞幀
        if rel_img in bad_frames:
            continue

        is_suspicious = img_name in suspicious_frames
        bbox_json = os.path.join(root, file)

        tasks.append({
            'img_path': img_path,
            'bbox_json': bbox_json,
            'rel_img': rel_img,
            'is_suspicious': is_suspicious
        })

# 可疑幀排前面
tasks.sort(key=lambda x: (not x['is_suspicious'], x['rel_img']))

print(f"共 {len(tasks)} 張圖片待確認（可疑幀排在前面）")

for i, task in enumerate(tasks):
    img = cv2.imread(task['img_path'])
    if img is None:
        continue

    h, w = img.shape[:2]

    # 畫 bbox
    with open(task['bbox_json'], 'r') as f:
        bboxes = json.load(f)

    for b in bboxes:
        if isinstance(b, dict):
            bbox = b['bbox']
            track_id = b.get('track_id', -1)
            conf = b.get('conf', 0)
        else:
            bbox = b
            track_id = -1
            conf = 0

        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = (0, 255, 0)  # 正常幀綠色
        if task['is_suspicious']:
            color = (0, 0, 255)  # 可疑幀紅色

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"ID:{track_id} {conf:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, color, 2
        )

    # 顯示狀態資訊
    status = "⚠️ 可疑幀" if task['is_suspicious'] else "✅ 正常幀"
    cv2.putText(
        img,
        f"{status} [{i+1}/{len(tasks)}]",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 255, 255), 2
    )
    cv2.putText(
        img,
        task['rel_img'],
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 255, 255), 1
    )
    cv2.putText(
        img,
        "d=刪除  k=保留  q=離開",
        (10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 0), 2
    )

    # 顯示圖片
    # 縮放視窗避免太大
    display_h = min(h, 800)
    display_w = int(w * display_h / h)
    display = cv2.resize(img, (display_w, display_h))

    cv2.imshow('YOLO 視覺化確認', display)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('d'):
        bad_frames.add(task['rel_img'])
        # 即時寫入 confirmed_bad.txt
        with open(CONFIRMED_BAD, 'a') as f:
            f.write(task['rel_img'] + '\n')
        print(f"❌ 刪除：{task['rel_img']}")

    elif key == ord('k'):
        print(f"✅ 保留：{task['rel_img']}")

    elif key == ord('q'):
        print("離開！")
        break

cv2.destroyAllWindows()
print(f"\n完成！共標記 {len(bad_frames)} 張壞幀")
print(f"請執行刪除腳本處理 {CONFIRMED_BAD}")
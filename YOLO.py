from ultralytics import YOLO
import os
import json

MODEL_PATH = "yolov8n-pose.pt"
IMAGE_DIR = "版本1/dataset/images"
OUT_DIR = "版本1/annotations/yolo_boxes"
CONF_TH = 0.3
BBOX_EXPAND = 1.25

os.makedirs(OUT_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)


def expand_bbox(bbox, scale=1.25):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    return [
        max(cx - w / 2, 0),
        max(cy - h / 2, 0),
        cx + w / 2,
        cy + h / 2
    ]


for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    results = model.predict(source=img_path, save=False, verbose=False)

    bboxes = []

    for r in results:
        if r.boxes is None:
            continue

        for box, cls, conf in zip(
            r.boxes.xyxy,
            r.boxes.cls,
            r.boxes.conf
        ):
            if int(cls) != 0:        # person class
                continue
            if conf < CONF_TH:
                continue

            bbox = box.tolist()
            bbox = expand_bbox(bbox, BBOX_EXPAND)
            bboxes.append(bbox)

    out_path = os.path.join(
        OUT_DIR, img_name.replace(".jpg", ".json")
    )
    with open(out_path, "w") as f:
        json.dump(bboxes, f)

    print(f"{img_name}: {len(bboxes)} persons")

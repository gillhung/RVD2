# real_virtual_dancing
special project

input:影片

Step 1  影片 → 影格
Step 2  YOLO 偵測人物（bbox）
Step 3  MediaPipe Pose（33 keypoints，pseudo-label）
Step 4  深度估計（MiDaS / DepthAnything）
Step 5  Keypoints × Depth（關節深度對齊）
Step 6  利用深度做標註清洗 / 修正
Step 7  MediaPipe 33 → COCO 17
Step 8  建立 COCO Dataset
Step 9  訓練 A-HRNet

output: 一個畫面標記多人

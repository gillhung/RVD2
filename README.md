# real_virtual_dancing
special project

input:影片

Step 1  影片 → 影格

Step 2  YOLO 偵測人物（bbox）

Step 3  bounding box JSON.py 切單人

Step 4  MediaPipe Pose（33 keypoints，pseudo-label）

Step 5  建立 COCO Dataset

Step 6  MediaPipe 33 → COCO 17

Step 7  keypoint_image.py 

Step 8  訓練 A-HRNet

output: 有標記點的 圖片 & json

標記圖片: annotations/marked_images

標記json(17點): annotations/coco_annotations.json

標記json(33點): annotations/mediapipe_pose

切割完的照片(未標記): dataset/images

切割完的單人照片: dataset/single_person_images

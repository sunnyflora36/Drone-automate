import os
from ultralytics import YOLOv10

YOLOV10_P2_CONFIG = """
nc: 1
scales:
  m: [0.67, 0.75, 768]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]          # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]         # 1-P2/4
  - [-1, 3, C2f, [128, True]]          # 2-P2/4
  - [-1, 1, Conv, [256, 3, 2]]         # 3-P3/8
  - [-1, 6, C2f, [256, True]]          # 4-P3/8
  - [-1, 1, SCDown, [512, 3, 2]]       # 5-P4/16
  - [-1, 6, C2f, [512, True]]          # 6-P4/16
  - [-1, 1, SCDown, [1024, 3, 2]]      # 7-P5/32
  - [-1, 3, C2fCIB, [1024, True]]      # 8-P5/32
  - [-1, 1, SPPF, [1024, 5]]           # 9
  - [-1, 1, PSA, [1024]]               # 10-P5/32

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 11
  - [[-1, 6], 1, Concat, [1]]                  # 12
  - [-1, 3, C2fCIB, [512, True]]               # 13

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 14
  - [[-1, 4], 1, Concat, [1]]                  # 15
  - [-1, 3, C2f, [256]]                        # 16

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 17
  - [[-1, 2], 1, Concat, [1]]                  # 18
  - [-1, 3, C2f, [128]]                        # 19

  - [-1, 1, Conv, [128, 3, 2]]                 # 20
  - [[-1, 16], 1, Concat, [1]]                 # 21
  - [-1, 3, C2f, [256]]                        # 22

  - [-1, 1, SCDown, [256, 3, 2]]               # 23
  - [[-1, 13], 1, Concat, [1]]                 # 24
  - [-1, 3, C2fCIB, [512, True]]               # 25

  - [-1, 1, SCDown, [512, 3, 2]]               # 26
  - [[-1, 10], 1, Concat, [1]]                 # 27
  - [-1, 3, C2fCIB, [1024, True]]              # 28

  - [[19, 22, 25, 28], 1, v10Detect, [nc]]     # 29
"""


def train_yolov10():
    yaml_filename = "yolov10m_p2_drone.yaml"

    with open(yaml_filename, "w", encoding="utf-8") as f:
        f.write(YOLOV10_P2_CONFIG.strip())

    model = YOLOv10(yaml_filename)
    model.load("yolov10m.pt")

    model.train(
        data="drone_data.yaml",
        epochs=50,
        imgsz=1024,
        batch=4,
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3.0,
        optimizer="AdamW",
        weight_decay=0.0005,
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=15,
        device=0,
        project="drone_yolov10_project",
        name="yolov10m_p2_drone_run",
    )

    metrics = model.val(imgsz=1024)
    print(f"Validation mAP50-95: {metrics.box.map}")


if __name__ == "__main__":
    train_yolov10()

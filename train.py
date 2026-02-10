from ultralytics import YOLO

model = YOLO("/mnt/mintuser-home/mint-profile/Desktop/dusk-dev/runs/detect/runs/train/captcha_test9/weights/best.pt")

model.train(
    data="data.yaml",
    epochs=40,
    batch=16,
    imgsz=384,
    workers=0,
    project="runs/train",
    name="captcha_test_stabilise",
    patience=10,
    fliplr=0.0,
    flipud=0.0,
)



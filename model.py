from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(data='mydata.yaml', epochs=10)

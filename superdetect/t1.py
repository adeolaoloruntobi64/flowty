from ultralytics import YOLO

# Maybe in the future, if I really want to, I'll label all 417 images
# however, I am content with what I have right now
model = YOLO("models/yolo26s-seg.pt")
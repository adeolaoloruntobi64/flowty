from ultralytics import YOLO

# Maybe in the future, if I really want to, I'll label all 417 images
# however, I am content with what I have right now
# Flow free shapes party pack lvl 53
# Flow free bridges variery pack lvl 10
# Those are some weird ones my cv algo can't do rn
model = YOLO("models/yolo26s-seg.pt")
from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated

model = YOLO(r'C:\Users\alexs\Desktop\cvr_october_29_30\runs\detect\train20\weights\best.pt')
cap = cv2.VideoCapture(0)

cap.set(4, 640) #cap.set(3, 640)
cap.set(3, 480) #cap.set(4, 480)

while True:
    _, img = cap.read()
    _, img = cap.read()

    # BGR to RGB conversion is performed under the hood
    # see: https://github.com/ultralytics/ultralytics/issues/2575
    results = model.predict(img, conf=0.55)

    for r in results:

        annotator = Annotator(img)

        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])

    img = annotator.result()
    cv2.imshow('YOLO V8 Detection', img)
    print(img)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllцWindows()

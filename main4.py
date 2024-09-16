import cv2
from ultralytics import YOLO
import pandas as pd
from tracker import Tracker
import numpy as np
import cvzone

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

model = YOLO("best.pt")  

cap = cv2.VideoCapture('chicks.mp4')

# Assuming the video is in 9:16 aspect ratio
frame_height = 600
frame_width = int(frame_height * (9 / 16))

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")
count = 0
tracker = Tracker()
chickscount = []
cy1 = int(frame_height * (1 / 4))  # Adjusted for the 9:16 frame height
offset = 6

while True:
    ret, frame = cap.read()
    
    count += 1
    if count % 3 != 0:
        continue
    if not ret:
        break
    
    frame = cv2.resize(frame, (frame_width, frame_height))

    results = model(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    bbox_list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        
        d = int(row[5])
        if d < len(class_list):
            c = class_list[d]
        else:
            print(f'Warning: Index {d} is out of range for class list.')
            continue
        
        bbox_list.append([x1, y1, x2, y2])
        
    tracked_bboxes = tracker.update(bbox_list)
    for bbox in tracked_bboxes:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
            if id not in chickscount:
                chickscount.append(id)
    
    counting = len(chickscount)
    cvzone.putTextRect(frame, f'{counting}', (50, 60), 2, 2)
#    cv2.line(frame, (5, cy1), (frame_width - 5, cy1), (255, 0, 255), 2)
    cv2.imshow("RGB", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

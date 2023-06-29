# myobj_simple.py
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from MediapipeObjectDetection import MediapipeObjectDetection as ObjDetection

cap = cv2.VideoCapture(0)
Obj = ObjDetection(score_threshold=0.5)
while cap.isOpened():
    ret, frame = cap.read()
    Obj.detect(frame)
    print(Obj.num_detected_objects)
    annotated_frame = Obj.visualize(frame)
    cv2.imshow('annotated frame', annotated_frame)
    key = cv2.waitKey(1)&0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
Obj.release()
cap.release()
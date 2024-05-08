# myhand_ges_simple.py
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from MediapipeHandGestureRecognition import MediapipeHandGestureRecognition as HandGesRec

cap = cv2.VideoCapture(0)
HandGes = HandGesRec()
while cap.isOpened():
    ret, frame = cap.read()
    HandGes.detect(frame)
    if HandGes.num_detected_hands>0:
        print(HandGes.get_gesture(0), HandGes.get_score_gesture(0))
    annotated_frame = HandGes.visualize(frame)
    cv2.imshow('annotated frame', annotated_frame)
    key = cv2.waitKey(1)&0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
HandGes.release()
cap.release()

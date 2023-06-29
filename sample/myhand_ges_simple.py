# myhand_ges_simple.py
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from MediapipeHandGestureRecognition import MediapipeHandGestureRecognition as HandGesRec

cap = cv2.VideoCapture(0)
HandGes = HandGesRec()
while cap.isOpened():
    ret, frame = cap.read()
    flipped_frame = cv2.flip(frame, 1)
    HandGes.detect(flipped_frame)
    if HandGes.num_detected_hands>0:
        print(HandGes.get_gesture(0), HandGes.get_score_gesture(0))
    annotated_frame = HandGes.visualize(flipped_frame)
    cv2.imshow('annotated frame', annotated_frame)
    key = cv2.waitKey(1)&0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
HandGes.release()
cap.release()

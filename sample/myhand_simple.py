# myhand_simple.py
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from MediapipeHandLandmark import MediapipeHandLandmark as HandLmk

cap = cv2.VideoCapture(0)
Hand = HandLmk()
while cap.isOpened():
    ret, frame = cap.read()
    flipped_frame = cv2.flip(frame, 1)
    Hand.detect(flipped_frame)
    print(Hand.num_detected_hands)
    annotated_frame = Hand.visualize(flipped_frame)
    cv2.imshow('annotated frame', annotated_frame)
    key = cv2.waitKey(1)&0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
Hand.release()
cap.release()
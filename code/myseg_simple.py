# myseg_simple.py
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from MediapipeImageSegmentation import MediapipeImageSegmentation as ImgSeg

cap = cv2.VideoCapture(0)
Seg = ImgSeg()
while cap.isOpened():
    ret, frame = cap.read()
    Seg.detect(frame)
    normalized_masks = Seg.get_normalized_masks()
    cv2.imshow('multiclass mask', cv2.applyColorMap(normalized_masks, cv2.COLORMAP_JET))
    face_skin_masks = Seg.get_segmentation_mask(Seg.FACE_SKIN)
    cv2.imshow('face skin', face_skin_masks)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)&0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
Seg.release()
cap.release()
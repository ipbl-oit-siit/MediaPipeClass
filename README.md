# mediapipe_samples
- <mediapipe 0.10.0.0>
- [MediaPipe Examples](https://developers.google.com/mediapipe/solutions/examples)

---
## specification of mediapipe and our mediapipe class
- [ObjectDetecion](./ObjectDetecion.md)
- [HandLandmark](./HandLandmark.md)
- [FaceDetection](./FaceDetection.md)
- [FaceLandmark](./FaceLandmark.md)
- [HandLandmark](./HandLandmark.md)
- [HandGestureRecognition](./HandGestureRecognition.md)
- [ImageSegmentation](./ImageSegmentation.md)
- [PoseLandmark](./PoseLandmark.md)

---
## how to use
- ObjectDetection
```python
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
```
- HandLandmark
```python
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
```
- HandGestureRecognition
```python
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
```
- ImageSegmentation
```python
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
```
- FaceDetection
```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from MediapipeFaceDetection import MediapipeFaceDetection as FaceDect

cap = cv2.VideoCapture(0)
Face = FaceDect()
while cap.isOpened():
    ret, frame = cap.read()
    flipped_frame = cv2.flip(frame, 1)
    Face.detect(flipped_frame)
    annotated_frame = Face.visualize(flipped_frame)
    cv2.imshow('frame', annotated_frame)
    key = cv2.waitKey(1)&0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
Face.release()
cap.release()
```
- FaceLandmark
```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from MediapipeFaceLandmark import MediapipeFaceLandmark as FaceLmk

cap = cv2.VideoCapture(0)
Face = FaceLmk()
while cap.isOpened():
    ret, frame = cap.read()
    flipped_frame = cv2.flip(frame, 1)
    Face.detect(flipped_frame)
    annotated_frame = Face.visualize(flipped_frame)
    cv2.imshow('frame', annotated_frame)
    key = cv2.waitKey(1)&0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
Face.release()
cap.release()
```
- MediapipePoseLandmark
```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from MediapipePoseLandmark import MediapipePoseLandmark as PoseLmk

cap = cv2.VideoCapture(0)
Pose = PoseLmk()
while cap.isOpened():
    ret, frame = cap.read()
    Pose.detect(frame)
    masks = Pose.get_all_segmentation_masks()
    masked_frame = Pose.visualize_mask(frame, masks)
    annotated_frame = Pose.visualize(masked_frame)
    cv2.imshow('frame', annotated_frame)
    key = cv2.waitKey(1)&0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
Pose.release()
cap.release()
```

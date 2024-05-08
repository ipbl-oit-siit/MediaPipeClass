[Portal Site](https://github.com/ipbl-oit-siit/portal/tree/main) / [image_recognition Site](https://github.com/ipbl-oit-siit/image_recognition/tree/main)

# MediaPipe Class for iPBL
- <mediapipe 0.10.0.0> or later
- [MediaPipe Examples](https://developers.google.com/mediapipe/solutions/examples)

<image src="image/mediapipe.jpg" width=50%>

---
## specification of mediapipe and our mediapipe class
- [PoseLandmark](docs/PoseLandmark.md)
- [Hands](docs/HandLandmark_and_GestureRecognition.md)
    - [HandLandmark](docs/HandLandmark_and_GestureRecognition.md#MediapipeHandLandmark)
    - [HandGestureRecognition](docs/HandLandmark_and_GestureRecognition.md#mediapipehandgesturerecognition)
- [FaceLandmark](docs/FaceLandmark.md)
- [FaceDetection](docs/FaceDetection.md)
- [ObjectDetecion](docs/ObjectDetection.md)
- [ImageSegmentation](docs/ImageSegmentation.md)

||Pose|Hand|HandGes|Face|FaceDtc|Obj|Seg|
|-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|**image to use**|frame|frame|frame|flipped_frame|flipped_frame|either|either|
|**`num_detected_*`**|o|o|o|o|o|o|-|
|**`num_landmarks`**|o|o|o|o|o|-|-|
|**coordinate data**|get_landmark|get_landmark|get_landmark|get_landmark|get_landmark<br>get_bounding_box|get_bounding_box|-|
|**optional**|`segmentation_mask`|`handedness`|`handedness`<br>`gesture_name`|-|-|`category_name`|`segmentation_mask`<br>`confidence_mask`|

---
## :red_square: Learning Tasks
1. The following `how to use` is a simple description of use of our class. Let's try to run them.
1. Next, go to the following page, which outlines the detailed usage of each of them, and execute them all.
    1. :o:[HOW2USE(more)](docs/how2use_more.md) :arrow_left: `Next`

---
## :red_square: how to use
- The following is a simple description of use of our class.
    - **:exclamation: Note that these programs must be placed in the same directory as `our Mediapipe Class file` to work.**
        - [MediaPipeClass.zip](./MediaPipeClass.zip "download")
    - Create a new python file in the same directory as `our Mediapipe Class file`, copy and paste the following sample code, and run it.
### :o: HandLandmark
- draw hand landmarks and handedness, and print the number of detected hands<br>
    <image src="image/myhand_simple.jpg" width=25% height=25%>
```python
# myhand_simple.py
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from MediapipeHandLandmark import MediapipeHandLandmark as HandLmk

cap = cv2.VideoCapture(0)
Hand = HandLmk()
while cap.isOpened():
    ret, frame = cap.read()
    Hand.detect(frame)
    print(Hand.num_detected_hands)
    annotated_frame = Hand.visualize(frame)
    cv2.imshow('annotated frame', annotated_frame)
    key = cv2.waitKey(1)&0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
Hand.release()
cap.release()
```
### :o: HandGestureRecognition
- draw hand landmarks and handedness, and print gesture name and its score
    - recognizable gesture: `None`, `Closed_Fist`, `Open_Plam`, `Pointing_up`, `Thumb_Down`, `Thumb_Up`, `Victory`, `ILoveYou`<br>
    <image src="image/myhand_ges_simple.jpg" width=25% height=25%>
```python
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
```
### :o: PoseLandmark
- draw pose landmarks and segmentation mask on image<br>
    <image src="image/mypose_simple.jpg" width=25% height=25%>
```python
# mypose_simple.py
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
### :o: FaceLandmark
- draw face landmarks<br>
    <image src="image/myface_simple.jpg" width=25% height=25%>
```python
# myface_simple.py
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
### :o: FaceDetection
- draw face bounding box, face keypoints, and detection score<br>
    <image src="image/myface_dtc_simple.jpg" width=25% height=25%>
```python
# myface_dtc_simple.py
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
### :o: ObjectDetection
- draw the all object's bounding box, object name and detection score, and print the number of detected objects<br>
    <image src="image/myobj_simple.jpg" width=25% height=25%>
```python
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
```
### :o: ImageSegmentation
- draw normalized segmentation masks, face skin mask, input image
    - dark blue:`background`, blue: `hair`, light blue: `body_skin`, yellow: `face_skin`, orange: `clothes`, red: `others`<br>
    <image src="image/myseg_simple.jpg" width=75%>
```python
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
```

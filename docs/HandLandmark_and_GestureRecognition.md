# mediapipe_samples
- <mediapipe 0.10.0.0>
- [MediaPipe Examples](https://developers.google.com/mediapipe/solutions/examples)


## MediapipeHandLandmark
### class variable
- model information
  - `base_url`, `model_name`, `model_folder_path`
- visualize params
  - e.g. `RIGHT_HAND_COLOR`, `FONT_SIZE`, ...
- hand landmark id
  - e.g. `WRIST = 0`, `THUMB_CMC = 1`
### instance variable
- `num_detected_hands`: number of detected objects
- (`detector`: mediapipe detector)
- (`results`: mediapipe detector's results)
### method
- `__init__( arguments are optional )`: constructor
  - arguments
    - `model_folder_path`: If you want to change the model folder path
    - `base_url`: If you want to change the model
    - `model_name`: If you want to change the model
    - `num_hands`: The maximum number of hands detected by the Hand landmark detector
      - Value Range: Any integer `> 0`
      - Default Value: `2`
    - `min_hand_detection_confidence`: The minimum confidence score for the hand detection to be considered successful in palm detection model.
      - Value Range: `0.0 - 1.0`
      - Default Value: `0.5`
    - `min_hand_presence_confidence`: The minimum confidence score for the hand presence score in the hand landmark detection model. In Video mode, if the hand presence confidence score from the hand landmark model is below this threshold, Hand Landmarker triggers the palm detection model. Otherwise, a lightweight hand tracking algorithm determines the location of the hand(s) for subsequent landmark detections.
      - Value Range: `0.0 - 1.0`
      - Default Value: `0.5`
    - `min_tracking_confidence`: The minimum confidence score for the hand tracking to be considered successful. This is the bounding box IoU threshold between hands in the current frame and the last frame. In Video mode and Stream mode of Hand Landmarker, if the tracking fails, Hand Landmarker triggers hand detection. Otherwise, it skips the hand detection.
      - Value Range: `0.0 - 1.0`
      - Default Value: `0.5`
- `detect( image )`
  - arguments
    - `image`: Input image (readed by cv2)
  - return values
    - `results`: Probably not necessary
- `get_normalized_landmark( id_hand, id_landmark )`
  - arguments
    - `id_hand`: ID number of the hand you want to get normalized landmark coordinate
    - `id_landmark`: ID number of the hand landmark you want to get normalized landmark coordinate
  - return values
    - `np.array([x, y, z])`: array of the coordinate
      - `x`: normalized x-coordinate, `y`: normallized y-coordinate, `z`: normalized z-coordinate
      - Value Range: `x:0.0-1.0`, `y:0.0-1.0`, `z:0.0-1.0`
- `get_landmark( id_hand, id_landmark )`
  - arguments
    - `id_hand`: ID number of the hand you want to get landmark coordinate
    - `id_landmark`: ID number of the hand landmark you want to get landmark coordinate
  - return values
    - `np.array([x, y, z])`: array of the coordinate
      - `x`: x-coordinate, `y`: y-coordinate, `z`: z-coordinate
      - Value Range: `x:0-width`, `y:0-height`, `z:0-width`
- `category_name = get_handedness( id_hand )`
  - e.g. `Right`, `Left`
- `category_score = get_score_handedness( id_hand )`
- `annotated_image = visualize( image )`
  - `annotated_image`: Image with hand landmark points and category names for all detected hands on the input image
- `annotated_image = visualize_with_mp( image )`
  - mediapipe visualizing settings
- `release()`: Close mediapipe's `detector`
### how to use
- simple sample
```python
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
- other sample
  - You can see it in the main function of `MediapipeHandLandmark.py`


----
## MediapipeHandGestureRecognition
### inheritance
- `class MediapipeHandLandmark`
### class variable
- same of the `class MediapipeHandLandmark`
### instance variable
- `num_detected_hands`: number of detected objects
- (`recognizer`: mediapipe recognizer)
- (`results`: mediapipe recognizer's results)
### method
- `__init__( arguments are optional )`: constructor
  - arguments
    - same of the `class MediapipeHandLandmark`
- `detect( image )`
  - arguments
    - `image`: Input image (readed by cv2)
  - return values
    - `results`: Probably not necessary
- [inheritance] `get_normalized_landmark( id_hand, id_landmark )`
- [inheritance] `get_landmark( id_hand, id_landmark )`
- [inheritance] `category_name = get_handedness( id_hand )`
- [inheritance] `category_score = get_score_handedness( id_hand )`
- `gesture_name = get_gesture( id_hand )`
  - e.g. `victory`, `thumbs up`
- `gesture_score = get_gesture( id_hand )`
- [inheritance] `annotated_image = visualize( image )`
- [inheritance] `annotated_image = visualize_with_mp( image )`
- `release()`: Close mediapipe's `recognizer`
### how to use
- simple sample
```python
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
- other sample
  - You can see it in the main function of `MediapipeHandGestureRecognition.py`


---
## specification of Mediapipe's `results`
### hand_landmarker
- https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- ***examples of how to reference result data `results`***
  - normalized x-coordinate the j-th landmark of the i-th hand<br>
    `results.hand_landmarks[i][j].x`
  - x-coordinate the j-th landmark of the i-th hand<br>
    `int(results.hand_landmarks[i][j].x * width)`
  - category_name (e.g.`Right`) of the i-th hand<br>
    `results.handedness[i][0].category_name`
- ***data structure of result***
  - results
    - hand_landmarks (z-cordinate is based on 0-th landmark `wrist`)
      - 0: (hand_id)
        - 0: (landmark_id)
          - x
          - y
          - z
          - presence
          - visibility
        - 1: (landmark_id)
          - ...
      - 1: (hand_id)
        - ...
    - handedness
      - 0: (hand_id)
        - 0:
          - index
          - category_name
          - display_name
          - score
      - 1: (hand_id)
        - ...
    - hand_world_landmarks (representing real-world 3D coordinates in meters with the origin at the hand’s geometric center)
      - 0: (hand_id)
        - 0: (landmark_id)
          - x
          - y
          - z
          - presence
          - visibility
        - 1: (landmark_id)
          - ...
      - 1: (hand_id)
        - ...

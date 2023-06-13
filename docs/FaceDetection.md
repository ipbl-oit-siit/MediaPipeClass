# mediapipe_samples
- <mediapipe 0.10.0.0>
- [MediaPipe Examples](https://developers.google.com/mediapipe/solutions/examples)


----
## MediapipeFaceDetection
### class variable
- model information
  - `base_url`, `model_name`, `model_folder_path`
- visualize params
  - e.g. `TEXT_COLOR`, `FONT_SIZE`, ...
- blaze_face_short_range's id
  - e.g. `LEFT_EYE=0`, `RIGHT_EYE=1`
### instance variable
- `num_detected_faces`: number of detected faces
- (`detector`: mediapipe detector)
- (`results`: mediapipe detector's results)
### method
- `__init__( arguments are optional )`: constructor
  - arguments
    - `model_folder_path`: If you want to change the model folder path
    - `base_url`: If you want to change the model
    - `model_name`: If you want to change the model
    - `min_detection_confidence`: The minimum confidence score for the face detection to be considered successful.
      - Value Range: `0.0 - 1.0`
      - Default Value: `0.5`
    - `min_suppression_confidence`: The minimum non-maximum-suppression threshold for face detection to be considered overlapped.
      - Value Range: `0.0 - 1.0`
      - Default Value: `0.3`
- `detect( image )`
  - arguments
    - `image`: Input image (readed by cv2)
      - Input image is a frame image flipped holizontal! Otherwise, left eye and right eye are reversed.
  - return values
    - `results`: Probably not necessary
- `get_normalized_landmark( id_face, id_keypoint )`
  - arguments
    - `id_face`: ID number of the face you want to get normalized landmark coordinate
    - `id_keyporint`: ID number of the face keypoint you want to get normalized landmark coordinate
  - return values
    - `np.array([x, y])`: array of the coordinate
      - `x`: normalized x-coordinate, `y`: normallized y-coordinate
      - Value Range: `x:0.0-1.0`, `y:0.0-1.0`
- `get_landmark( id_face, id_keypoint )`
  - arguments
    - `id_face`: ID number of the face you want to get landmark coordinate
    - `id_keypoint`: ID number of the face keypoint you want to get landmark coordinate
  - return values
    - `np.array([x, y])`: array of the coordinate
      - `x`: x-coordinate, `y`: y-coordinate
      - Value Range: `x:0-width`, `y:0-height`
- `score = get_score( id_face )`
  - detection confidence score
- `annotated_image = visualize( image )`
  - `annotated_image`: Image with face RoI and face keypoints for all detected faces on the input image
- `release()`: Close mediapipe's `detector`
### how to use
- simple sample
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
- other sample
  - You can see it in the main function of `MediapipeFaceDetection.py`



## specification of Mediapipe's `results`
### face_detecion (not yet)
- https://developers.google.com/mediapipe/solutions/vision/face_detector













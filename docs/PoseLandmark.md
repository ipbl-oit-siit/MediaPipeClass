# mediapipe_samples
- <mediapipe 0.10.0.0>
- [MediaPipe Examples](https://developers.google.com/mediapipe/solutions/examples)


----
## MediapipePoseLandmark
### class variable
- model information
  - `base_url`, `model_name`, `model_folder_path`
- visualize params
  - e.g. `FONT_COLOR`, `FONT_SIZE`, ...
- pose landmark id
  - e.g. `NOSE=0`, `LEFT_EYE_INNER=1`
### instance variable
- `num_detected_poses`: number of detected poses
- (`detector`: mediapipe detector)
- (`results`: mediapipe detector's results)
### method
- `__init__( arguments are optional )`: constructor
  - arguments
    - `model_folder_path`: If you want to change the model folder path
    - `base_url`: If you want to change the model
    - `model_name`: If you want to change the model
    - `num_poses`:  The maximum number of poses detected by the Pose landmark detector
      - Value Range: Any integer `> 0`
      - Default Value: `2`
    - `min_pose_detection_confidence`: The minimum confidence score for the pose detection to be considered successful.
      - Value Range: `0.0 - 1.0`
      - Default Value: `0.5`
    - `min_pose_presence_confidence`: The minimum confidence score of pose presence score in the pose landmark detection.
      - Value Range: `0.0 - 1.0`
      - Default Value: `0.5`
    - `min_tracking_confidence`: The minimum confidence score for the pose tracking to be considered successful.
      - Value Range: `0.0 - 1.0`
      - Default Value: `0.5`
    - `output_segmentation_masks`: Whether Pose Landmarker outputs a segmentation mask for the detected pose.
      - Default Value: `True`
- `detect( image )`
  - arguments
    - `image`: Input image (readed by cv2)
  - return values
    - `results`: Probably not necessary
- `get_normalized_landmark( id_pose, id_landmark )`
  - arguments
    - `id_pose`: ID number of the pose you want to get normalized landmark coordinate
    - `id_landmark`: ID number of the pose landmark you want to get normalized landmark coordinate
  - return values
    - `np.array([x, y, z])`: array of the coordinate
      - `x`: normalized x-coordinate, `y`: normallized y-coordinate, `z`: normallized z-coordinate
      - Value Range: `x:0.0-1.0`, `y:0.0-1.0`, `z:0.0-1.0`
- `get_landmark( id_pose, id_landmark )`
  - arguments
    - `id_pose`: ID number of the pose you want to get landmark coordinate
    - `id_landmark`: ID number of the pose landmark you want to get landmark coordinate
  - return values
    - `np.array([x, y, z])`: array of the coordinate
      - `x`: x-coordinate, `y`: y-coordinate, `z`: z-coordinate
      - Value Range: `x:0-width`, `y:0-height`, `z:0-width`
- `visibility_score = get_landmark_visibility( id_pose, id_landmark )`
  - If this score is low, you should not use its landmark.
- `presence_score = get_landmark_presence( id_pose, id_landmark )`
  - If this score is low, you should not use its landmark.
- `segmentated_mask = get_segmentation_mask( id_pose )`
  - `segmentated_mask`
    - Type: `np.ndarray()`
    - Range: `[0, 1]`
- `all_segmentated_masks = get_all_segmentation_mask()`
  - `all_segmentated_masks`
    - Logical OR of all segmentation masks
    - Type: `np.ndarray()`
    - Range: `0-255`
- `masked_image = visualize_mask( image, mask )`
  - `masked_image`: masked image by using input `mask`
- `annotated_image = visualize( image )`
  - `annotated_image`: Image with pose landmarks for all detected poses on the input image
- `annotated_image = visualize_with_mp( image )`
  - mediapipe visualizing settings
- `release()`: Close mediapipe's `detector`
### how to use
- simple sample
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
- other sample
  - You can see it in the main function of `MediapipePoseLandmark.py`


----
## specification of Mediapipe's `results`
### pose_estimation
- https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
- ***examples of how to reference result data `results`***
  - normalized x-coordinate the j-th landmark of the i-th pose<br>
    `results.pose_landmarks[i][j].x`
  - x-coordinate the j-th landmark of the i-th pose<br>
    `int(results.pose_landmarks[i][j].x * width)`
  - segmentation_mask of the i-th pose<br>
    `results.segmentation_masks[i].numpy_view()`
- ***data structure of result***
  - results
    - pose_landmarks
      - 0: (pose_id)
        - 0: (landmark_id)
          - x
          - y
          - z
          - presence
          - visibility
        - 1: (landmark_id)
          - ...
      - 1: (pose_id)
        - ...
    - segmentation_masks
      - 0: (pose_id) (mediapipe image)
      - 1: (pose_id) (mediapipe image)
        - ...
    - pose_world_landmarks
      - 0: (pose_id)
        - 0: (landmark_id)
          - x
          - y
          - z
          - presence
          - visibility
        - 1: (landmark_id)
          - ...
      - 1: (pose_id)
        - ...

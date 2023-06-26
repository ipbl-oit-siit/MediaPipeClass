# mediapipe_samples
- <mediapipe 0.10.0.0>
- [MediaPipe Examples](https://developers.google.com/mediapipe/solutions/examples)



----
## MediapipeFaceLandmark
### class variable
- model information
  - `base_url`, `model_name`, `model_folder_path`
- visualize params
  - e.g. `FONT_COLOR`, `FONT_SIZE`, ...
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
    - `num_faces`:  The maximum number of hands detected by the Hand landmark detector
      - Value Range: Any integer `> 0`
      - Default Value: `2`
    - `min_face_detection_confidence`: The minimum confidence score for the face detection to be considered successful.
      - Value Range: `0.0 - 1.0`
      - Default Value: `0.5`
    - `min_face_presence_confidence`: The minimum confidence score of face presence score in the face landmark detection.
      - Value Range: `0.0 - 1.0`
      - Default Value: `0.5`
    - `min_tracking_confidence`: The minimum confidence score for the face tracking to be considered successful.
      - Value Range: `0.0 - 1.0`
      - Default Value: `0.5`
    - `output_face_blendshapes`: Whether Face Landmarker outputs face blendshapes. Face blendshapes are used for rendering the 3D face model.
      - Default Value: `False`
    - `output_facial_transformation_matrixes`: Whether FaceLandmarker outputs the facial transformation matrix. FaceLandmarker uses the matrix to transform the face landmarks from a canonical face model to the detected face, so users can apply effects on the detected landmarks.
      - Default Value: `False`
- `detect( image )`
  - arguments
    - `image`: Input image (readed by cv2)
      - Input image is a frame image flipped holizontal! Otherwise, left and right of the face are reversed.
  - return values
    - `results`: Probably not necessary
- `get_normalized_landmark( id_face, id_landmark )`
  - arguments
    - `id_face`: ID number of the face you want to get normalized landmark coordinate
    - `id_landmark`: ID number of the face landmark you want to get normalized landmark coordinate
  - return values
    - `np.array([x, y, z])`: array of the coordinate
      - `x`: normalized x-coordinate, `y`: normallized y-coordinate, `z`: normallized z-coordinate
      - Value Range: `x:0.0-1.0`, `y:0.0-1.0`, `z:0.0-1.0`
- `get_landmark( id_face, id_landmark )`
  - arguments
    - `id_face`: ID number of the face you want to get landmark coordinate
    - `id_landmark`: ID number of the face landmark you want to get landmark coordinate
  - return values
    - `np.array([x, y, z])`: array of the coordinate
      - `x`: x-coordinate, `y`: y-coordinate, `z`: z-coordinate
      - Value Range: `x:0-width`, `y:0-height`, `z:0-width`
- `annotated_image = visualize( image )`
  - `annotated_image`: Image with face landmarks for all detected faces on the input image
- `annotated_image = visualize_with_mp( image )`
  - mediapipe visualizing settings
- `release()`: Close mediapipe's `detector`
### how to use
- simple sample
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
- other sample
  - You can see it in the main function of `MediapipeFaceLandmark.py`





## specification of Mediapipe's `results`
### face_landmarker
- https://developers.google.com/mediapipe/solutions/vision/face_landmarker
- ***examples of how to reference result data `results`***
  - normalized x-coordinate the j-th landmark of the i-th face<br>
    `results.face_landmarks[i][j].x`
  - x-coordinate the j-th landmark of the i-th face<br>
    `int(results.face_landmarks[i][j].x * width)`
- ***data structure of result***
  - results
    - face_landmarks (z-cordinate is based on 0-th landmark `wrist`)
      - 0: (face_id)
        - 0: (landmark_id)
          - x
          - y
          - z
          - presence
          - visibility
        - 1: (landmark_id)
          - ...
      - 1: (face_id)
        - ...
    - face_blandshapes
      - 0: (face_id)
        - 0: (blendshapes_idx)
          - index (`0`)
          - category_name (`_neutral`)
          - display_name
          - score
        - 1: (blendshapes_idx)
          - ...
      - 1: (face_id)
        - ...
    - facial_transformation_matrixes
      - 0: (face_id)
        - 0:
          - array([0:4])
        - 1:
        - 2:
        - 3:
      - 1: (face_id)
        - ...



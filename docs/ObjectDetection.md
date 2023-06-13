# mediapipe_samples
- <mediapipe 0.10.0.0>
- [MediaPipe Examples](https://developers.google.com/mediapipe/solutions/examples)



## ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) MediapipeObjectDetection
### class variable
- model information
  - `base_url`, `model_name`, `model_folder_path`
- visualize params
### instance variable
- `num_detected_objects`: number of detected objects
- (`detector`: mediapipe detector)
- (`results`: mediapipe detector's results)
### method
- `__init__( arguments are optional )`: constructor
  - arguments
    - `model_folder_path`: If you want to change the model folder path
    - `base_url`: If you want to change the model
    - `model_name`: If you want to change the model
    - `max_results`: Sets the optional maximum number of top-scored detection results to return.
      - Value Range: Any positive numbers
      - Default Value: `-1` (all results are returned)
    - `score_threshold`: Sets the prediction score threshold that overrides the one provided in the model metadata (if any). Results below this value are rejected.
      - Value Range: Any float `[0.0, 1.0]`
      - Default Value: `0.0` (all results are detected)
- `detect( image )`
  - arguments
    - `image`: Input image (readed by cv2)
  - return values
    - `results`: Probably not necessary
- `get_bounding_box( id_object )`
  - arguments
    - `id_object`: Number of the object you want to get bounding box
  - return values
    - `np.array([x, y, w, h])`: array of the bounding box information
      - `x`: x-coordinate, `y`: y-coordinate, `w`: width, `h`: height
- `category_name = get_category_name( id_object )`
- `category_score = get_category_score( id_object )`
- `annotated_image = visualize( image )`
  - `annotated_image`: Image with bounding boxes and category names for all detected objects on the input image
- `release()`: Close mediapipe's `detector`
### how to use
- simple sample
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
- other sample
  - You can see it in the main function of `MediapipeObjectDetection.py`

## ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) specification of Mediapipe's `results`
### object_detection
- https://developers.google.com/mediapipe/solutions/vision/object_detector
- ***examples of how to reference result data `results`***
  - x-coordinate of upper left point of the i-th object's bounding box<br>
    `results.detections[i].bounding_box.origin_x`
  - category_name (e.g.`parson`) of the i-th object<br>
    `results.detections[i].categories[0].category_name`
- ***data structure of result***
  - results
    - detections
      - 0:
        - bounding_box
          - origin_x
          - origin_y
          - width
          - height
        - categories
          - 0:
            - category_name
            - score
      - 1:
        - bounding_box
          - origin_x
          - origin_y
          - width
          - height
        - categories
          - 0:
            - category_name
            - score
      - ...




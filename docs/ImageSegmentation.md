# mediapipe_samples
- <mediapipe 0.10.0.0>
- [MediaPipe Examples](https://developers.google.com/mediapipe/solutions/examples)


----
## MediapipeImageSegmentation
### class variable
- model information
  - `base_url`, `model_name`, `model_folder_path`
- skin type id
  - e.g. `BACKGROUND=0`, `HAIR=1`
### instance variable
- (`segmenter`: mediapipe segmenter)
- (`results`: mediapipe segmenter's results)
### method
- `__init__( arguments are optional )`: constructor
  - arguments
    - `model_folder_path`: If you want to change the model folder path
    - `base_url`: If you want to change the model
    - `model_name`: If you want to change the model
    - `output_category_mask`: If set to True, the output includes a segmentation mask as a uint8 image, where each pixel value indicates the winning category value.
      - Default Value: `True`
    - `output_confidence_masks`: If set to True, the output includes a segmentation mask as a float value image, where each float value represents the confidence score map of the category.
      - Default Value: `True`
- `detect( image )`
  - arguments
    - `image`: Input image (readed by cv2)
  - return values
    - `results`: Probably not necessary
- `get_segmentation_masks()`
  - return values
    - `segmentation_mask`: pixel value is skin type id.
      - `BACKGROUND = 0`
      - `HAIR = 1`
      - `BODY_SKIN = 2`
      - `FACE_SKIN = 3`
      - `CLOTHES = 4`
      - `OTHERS = 5`
- `mask = get_segmentation_mask( skin_type )`
  - segmentation mask of the input skin type
    - skin type region: `255`
    - others: `0`
- `get_normalized_masks()`
  - for visualizing
  - return values
    - `normalized_masks`: pixel value is normalized into `0-255`.
- `release()`: Close mediapipe's `detector`
### how to use
- simple sample
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
- other sample
  - You can see it in the main function of `MediapipeImageSegmentation.py`
  - segmentation models: [MediaPipe Site](https://developers.google.com/mediapipe/solutions/vision/image_segmenter#models)
    - this sample uses `selfieMulticlass (256x256)` model





## specification of Mediapipe's `results`
### image_segmentation
- https://developers.google.com/mediapipe/solutions/vision/image_segmenter
- ***examples of how to reference result data `results`***
  - i-th segmented_mask (mediapipe image --> ndarray)<br>
    `results[0].numpy_view()`
- ***data structure of result***
  - results (mediapipe image)
      - 0:
        - height
        - width

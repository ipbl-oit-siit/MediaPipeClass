import os
import urllib.request
import time
import numpy as np
# https://github.com/opencv/opencv/issues/17687
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mediapipe as mp

# https://developers.google.com/mediapipe/solutions/vision/object_detector
class MediapipeObjectDetection():
    # https://storage.googleapis.com/mediapipe-models/
    base_url = 'https://storage.googleapis.com/mediapipe-tasks/object_detector/'
    model_name = 'efficientdet_lite0_fp32.tflite'
    model_folder_path = './models'

    H_MARGIN = 10  # pixels
    V_MARGIN = 30  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    TEXT_COLOR = (0, 255, 0)  # green

    # full list(efficientdet_lite0_fp32.tflite)
    # https://storage.googleapis.com/mediapipe-tasks/object_detector/labelmap.txt

    # https://developers.google.com/mediapipe/solutions/vision/object_detector
    def __init__(
            self,
            model_folder_path=model_folder_path,
            base_url=base_url,
            model_name=model_name,
            max_results=-1,
            score_threshold=0.0
            ):

        model_path = self.set_model(base_url, model_folder_path, model_name)
        options = mp.tasks.vision.ObjectDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            max_results=max_results,
            score_threshold=score_threshold,
            running_mode=mp.tasks.vision.RunningMode.VIDEO
        )
        self.detector = mp.tasks.vision.ObjectDetector.create_from_options(options)

    def set_model(self, base_url, model_folder_path, model_name):
        model_path = model_folder_path+'/'+model_name
        # download model if model file does not exist
        if not os.path.exists(model_path):
            # make directory if model_folder directory does not exist
            if not os.path.exists(model_folder_path):
                os.mkdir(model_folder_path)
            # download model file
            url = base_url+model_name
            save_name = model_path
            urllib.request.urlretrieve(url, save_name)
        return model_path

    def detect(self, image):
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
      self.results = self.detector.detect_for_video(mp_image, int(time.time() * 1000))
      self.num_detected_objects = len(self.results.detections)
      return self.results

    def get_bounding_box(self, id_object):
        if self.num_detected_objects == 0:
            print('no object')
            return None
        x = self.results.detections[id_object].bounding_box.origin_x
        y = self.results.detections[id_object].bounding_box.origin_y
        w = self.results.detections[id_object].bounding_box.width
        h = self.results.detections[id_object].bounding_box.height
        return np.array([x, y, w, h])

    def get_category_name(self, id_object) -> str:
        if self.num_detected_objects == 0:
            print('no object')
            return None
        return self.results.detections[id_object].categories[0].category_name

    def get_category_score(self, id_object) -> float:
        if self.num_detected_objects == 0:
            print('no object')
            return None
        return self.results.detections[id_object].categories[0].score

    def visualize(self, image):
        annotated_image = image.copy()
        for obj in self.results.detections:
            # 枠の左上座標，右下座標を算出し，描画する
            upper_left_point = (obj.bounding_box.origin_x, obj.bounding_box.origin_y)
            lower_right_point = (obj.bounding_box.origin_x+obj.bounding_box.width, obj.bounding_box.origin_y+obj.bounding_box.height)
            cv2.rectangle(annotated_image, upper_left_point, lower_right_point, self.TEXT_COLOR, thickness=self.FONT_THICKNESS)
            # 枠の左上あたりにカテゴリ名を表示する
            txt = obj.categories[0].category_name+'({:#.2f})'.format(obj.categories[0].score)
            lower_left_point_for_text = (obj.bounding_box.origin_x+self.H_MARGIN, obj.bounding_box.origin_y+self.V_MARGIN)
            cv2.putText(annotated_image, org=lower_left_point_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=self.FONT_SIZE, color=self.TEXT_COLOR, thickness=self.FONT_THICKNESS, lineType=cv2.LINE_4)
        return annotated_image

    def release(self):
        self.detector.close()

def main():
    cap = cv2.VideoCapture(0)
    Obj = MediapipeObjectDetection(score_threshold=0.5)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            print("Ignoring empty camera frame.")
            break

        results = Obj.detect(frame)

        if Obj.num_detected_objects > 0:
            index_object = 0 # object_index
            print(
                Obj.get_category_name(index_object),
                '(score:{:#.2f}):'.format(Obj.get_category_score(index_object)),
                Obj.get_bounding_box(index_object)
                )

        annotated_image = Obj.visualize(frame)

        cv2.imshow('annotated image', annotated_image)
        key = cv2.waitKey(1)&0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    Obj.release()
    cap.release()

if __name__=='__main__':
    main()

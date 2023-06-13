import os
import urllib.request
import time
import numpy as np
# https://github.com/opencv/opencv/issues/17687
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mediapipe as mp

# https://developers.google.com/mediapipe/solutions/vision/face_detector
class MediapipeFaceDetection():
    # https://storage.googleapis.com/mediapipe-models/
    base_url = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/'
    model_folder_path = './models'
    model_name = 'blaze_face_short_range.tflite'

    MARGIN = 10  # pixels
    ROW_SIZE = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    TEXT_COLOR = (255, 0, 0)  # red

    # blaze_face_short_range
    NUM_LMK = 6
    LEFT_EYE = 0
    RIGHT_EYE = 1
    NOSE_TIP = 2
    MOUTH = 3
    LEFT_EYE_TRAGION = 4
    RIGHT_EYE_TRAGION = 5

    def __init__(
            self,
            model_folder_path=model_folder_path,
            base_url=base_url,
            model_name=model_name,
            min_detection_confidence=0.5,
            min_suppression_threshold=0.3
            ):

        model_path = self.set_model(base_url, model_folder_path, model_name)
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            min_detection_confidence=min_detection_confidence,
            min_suppression_threshold=min_suppression_threshold,
            running_mode=mp.tasks.vision.RunningMode.VIDEO
        )
        self.detector = mp.tasks.vision.FaceDetector.create_from_options(options)
        self.num_landmarks = self.NUM_LMK # default

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

    def detect(self, img):
        self.size = img.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        self.results = self.detector.detect_for_video(mp_image, int(time.time() * 1000))
        self.num_detected_faces = len(self.results.detections)
        return self.results

    def get_normalized_landmark(self, id_face, id_keypoint):
        if self.num_detected_faces == 0:
            print('no face')
            return None
        x = self.results.detections[id_face].keypoints[id_keypoint].x
        y = self.results.detections[id_face].keypoints[id_keypoint].y
        return np.array([x, y])

    def get_landmark(self, id_face, id_keypoint):
        if self.num_detected_faces == 0:
            print('no face')
            return None
        height, width = self.size[:2]
        x = self.results.detections[id_face].keypoints[id_keypoint].x
        y = self.results.detections[id_face].keypoints[id_keypoint].y
        return np.array([int(x*width), int(y*height)])

    def get_bounding_box(self, id_face):
        x = self.results.detections[id_face].bounding_box.origin_x
        y = self.results.detections[id_face].bounding_box.origin_y
        w = self.results.detections[id_face].bounding_box.width
        h = self.results.detections[id_face].bounding_box.height
        return np.array([x, y, w, h])

    def get_score(self, id_face):
        if self.num_detected_faces == 0:
            print('no face')
            return None
        return self.results.detections[id_face].categories[0].score

    def visualize(self, img):
        annotated_image = img.copy()
        height, width, _ = img.shape
        for detection in self.results.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(annotated_image, start_point, end_point, self.TEXT_COLOR, 3)
            # Draw keypoints
            for keypoint in detection.keypoints:
                keypoint_px = [int(keypoint.x * width), int(keypoint.y*height)]
                cv2.circle(annotated_image, keypoint_px, 2, (0,255,0), 2)
                # Draw label and score
                category = detection.categories[0]
                category_name = category.category_name
                category_name = '' if category_name is None else category_name
                probability = round(category.score, 2)
                result_text = category_name + ' (' + str(probability) + ')'
                text_location = (self.MARGIN + bbox.origin_x, self.MARGIN + self.ROW_SIZE + bbox.origin_y)
                cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, self.FONT_SIZE, self.TEXT_COLOR, self.FONT_THICKNESS)
        return annotated_image

    def release(self):
        self.detector.close()

def main():
    cap = cv2.VideoCapture(0)
    FaceDet = MediapipeFaceDetection()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            print("Ignoring empty camera frame.")
            break

        # FaceDetection requires horizontal flip for input image
        flipped_frame = cv2.flip(frame, 1)

        results = FaceDet.detect(flipped_frame)

        if FaceDet.num_detected_faces > 0:
            index_face = 0
            index_keypoint = FaceDet.RIGHT_EYE # right eye
            print(
                'face score:{:#.2f}'.format(FaceDet.get_score(index_face)),
                'Right Eye:',
                FaceDet.get_normalized_landmark(index_face, index_keypoint),
                FaceDet.get_landmark(index_face, index_keypoint)
            )

        annotated_image = FaceDet.visualize(flipped_frame)

        cv2.imshow('annotated image', annotated_image)
        key = cv2.waitKey(1)&0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    FaceDet.release()
    cap.release()

if __name__=='__main__':
    main()

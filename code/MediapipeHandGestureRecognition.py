import os
import time
# https://github.com/opencv/opencv/issues/17687
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mediapipe as mp
from MediapipeHandLandmark import MediapipeHandLandmark

# https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer#get_started
class MediapipeHandGestureRecognition(MediapipeHandLandmark):
    # https://storage.googleapis.com/mediapipe-models/
    base_url = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/'
    model_name = 'gesture_recognizer.task'
    model_folder_path = './learned_models/mediapipe'

    # https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer#get_started
    def __init__(
            self,
            model_folder_path=model_folder_path,
            base_url=base_url,
            model_name=model_name,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            mode="image"
            ):
            # canned_gestures_classifier_options
            # custom_gestures_classifier_options

        self.mode = mode
        if self.mode == "image":
            rmode = mp.tasks.vision.RunningMode.IMAGE
        else:
            rmode = mp.tasks.vision.RunningMode.VIDEO

        model_path = self.set_model(base_url, model_folder_path, model_name)
        options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            # canned_gestures_classifier_options
            # custom_gestures_classifier_options
            running_mode=rmode
        )
        self.recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

    def detect(self, img):
        self.size = img.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if self.mode == "image":
            self.results = self.recognizer.recognize(mp_image)
        else:
            self.results = self.recognizer.recognize_for_video(mp_image, int(time.time() * 1000))
        self.num_detected_hands = len(self.results.hand_landmarks)
        return self.results

    def get_gesture(self, id_hand):
        if self.num_detected_hands == 0:
            print('no hand')
            return None
        return self.results.gestures[id_hand][0].category_name

    def get_score_gesture(self, id_hand):
        if self.num_detected_hands == 0:
            print('no hand')
            return None
        return self.results.gestures[id_hand][0].score

    def release(self):
        self.recognizer.close()


def main():
    cap = cv2.VideoCapture(0)
    GestRecog = MediapipeHandGestureRecognition()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            print("Ignoring empty camera frame.")
            break

        # HandGestureRecognition requires horizontal flip for input image
        results = GestRecog.detect(frame)

        if GestRecog.num_detected_hands > 0:
            index_hand = 0 #
            index_landmark = GestRecog.WRIST #
            print(
                GestRecog.get_handedness(index_hand),
                'score:{:#.2f}'.format(GestRecog.get_score_handedness(index_hand)),
                GestRecog.get_gesture(index_hand),
                'score:{:#.2f}'.format(GestRecog.get_score_gesture(index_hand)),
                GestRecog.get_normalized_landmark(index_hand, index_landmark),
                GestRecog.get_landmark(index_hand, index_landmark)
                )

        annotated_image = GestRecog.visualize(frame)

        cv2.imshow('annotated image', annotated_image)
        key = cv2.waitKey(1)&0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    GestRecog.release()
    cap.release()

if __name__=='__main__':
    main()

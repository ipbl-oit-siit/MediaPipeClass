import os
import urllib.request
import time
import numpy as np
# https://github.com/opencv/opencv/issues/17687
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#get_started
class MediapipePoseLandmark():
    # https://storage.googleapis.com/mediapipe-models/
    base_url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/'
    model_name = 'pose_landmarker_heavy.task'
    # base_url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/'
    # model_name = 'pose_landmarker_lite.task'
    model_folder_path = './models'

    # visualize params
    RADIUS_SIZE = 3  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = -1
    FONT_COLOR = (0, 255, 0)

    # pose landmark id
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

    def __init__(
            self,
            model_folder_path=model_folder_path,
            base_url=base_url,
            model_name=model_name,
            num_poses=2,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=True,
            ):

        model_path = self.set_model(base_url, model_folder_path, model_name)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            num_poses=num_poses,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=output_segmentation_masks,
            running_mode=mp.tasks.vision.RunningMode.VIDEO
        )
        self.detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def set_model(self, base_url, model_folder_path, model_name):
        model_path = model_folder_path+'/'+model_name
        # modelファイルが存在しない場合，ダウンロードしてくる
        if not os.path.exists(model_path):
            # model_folderが存在しない場合，フォルダを作成する
            if not os.path.exists(model_folder_path):
                os.mkdir(model_folder_path)
            # モデルをダウンロードする
            url = base_url+model_name
            save_name = model_path
            urllib.request.urlretrieve(url, save_name)
        return model_path

    def detect(self, img):
      self.size = img.shape
      # 画像データをmediapipe用に変換する
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

      # ポーズ検出を実行する
      self.results = self.detector.detect_for_video(mp_image, int(time.time() * 1000))
      self.num_detected_poses = len(self.results.pose_landmarks)

      return self.results

    def get_normalized_landmark(self, id_pose, id_landmark):
        if self.num_detected_poses == 0:
            print('no pose')
            return None
        height, width = self.size[:2]
        x = self.results.pose_landmarks[id_pose][id_landmark].x
        y = self.results.pose_landmarks[id_pose][id_landmark].y
        z = self.results.pose_landmarks[id_pose][id_landmark].z
        return np.array([x, y, z])

    def get_landmark(self, id_pose, id_landmark):
        if self.num_detected_poses == 0:
            print('no pose')
            return None
        height, width = self.size[:2]
        x = self.results.pose_landmarks[id_pose][id_landmark].x
        y = self.results.pose_landmarks[id_pose][id_landmark].y
        z = self.results.pose_landmarks[id_pose][id_landmark].z
        return np.array([int(x*width), int(y*height), int(z*width)])

    def get_landmark_visibility(self, id_pose, id_landmark):
        return self.results.pose_landmarks[id_pose][id_landmark].visibility

    def get_landmark_presence(self, id_pose, id_landmark):
        return self.results.pose_landmarks[id_pose][id_landmark].visibility

    def get_segmentation_mask(self, id_pose):
        if self.num_detected_poses == 0:
            print('no pose')
            return None
        return self.results.segmentation_masks[id_pose].numpy_view()

    def get_all_segmentation_masks(self):
        if self.num_detected_poses == 0:
            print('no pose')
            return None
        all_segmentation_masks = np.zeros_like(self.results.segmentation_masks[0], dtype=float)
        for mask in self.results.segmentation_masks:
            all_segmentation_masks = np.maximum(all_segmentation_masks, mask.numpy_view().astype(float))
        return (255*all_segmentation_masks).astype(np.uint8)

    def visualize_mask(self, img, mask):
        if self.results.segmentation_masks == None:
            print('no mask')
            return img
        segmentation_mask = mask.astype(float)/np.max(mask)
        visualized_mask = np.tile(segmentation_mask[:,:,None], [1,1,3])*0.7+0.3
        return (img * visualized_mask).astype(np.uint8)

    def visualize(self, img):
        annotated_image = np.copy(img)
        for i, pose in enumerate(self.results.pose_landmarks):
            for j in range(len(pose)):
                point = self.get_landmark(i, j)
                cv2.circle(annotated_image, tuple(point[:2]), self.RADIUS_SIZE, self.FONT_COLOR, thickness=self.FONT_THICKNESS)
        return annotated_image

    def visualize_with_mp(self, rgb_image):
        pose_landmarks_list = self.results.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
        return annotated_image

    def release(self):
        self.detector.close()


def main():
    cap = cv2.VideoCapture(0)
    Pose = MediapipePoseLandmark()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            print("Ignoring empty camera frame.")
            break

        results = Pose.detect(frame)

        if Pose.num_detected_poses > 0:
            index_pose = 0 #
            index_landmark = Pose.LEFT_WRIST # landmark
            print(
                'visibility:{:#.2f}'.format(Pose.get_landmark_visibility(index_pose, index_landmark)),
                'presence:{:#.2f}'.format(Pose.get_landmark_presence(index_pose, index_landmark)),
                Pose.get_normalized_landmark(index_pose, index_landmark),
                Pose.get_landmark(index_pose, index_landmark)
                )

        masks = Pose.get_all_segmentation_masks()
        mask_image = Pose.visualize_mask(frame, masks)
        annotated_image = Pose.visualize(mask_image)
        # annotated_image = Pose.visualize_with_mp(mask_image)

        cv2.imshow('annotated image', annotated_image)
        key = cv2.waitKey(1)&0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    Pose.release()
    cap.release()

if __name__=='__main__':
    main()

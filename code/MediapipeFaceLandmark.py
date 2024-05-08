from asyncio.windows_events import NULL
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

# https://developers.google.com/mediapipe/solutions/vision/face_landmarker
class MediapipeFaceLandmark():
    # https://storage.googleapis.com/mediapipe-models/
    base_url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/'
    model_name = 'face_landmarker.task'
    model_folder_path = './models'

    H_MARGIN = 10  # pixels
    V_MARGIN = 30  # pixels
    RADIUS_SIZE = 1  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    FONT_COLOR = (0, 255, 0)

    # face mesh landmarks
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    NUM_LMK = 478

    # blendshape landmarks
    # https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf

    # https://developers.google.com/mediapipe/solutions/vision/face_landmarker
    def __init__(
            self,
            model_folder_path=model_folder_path,
            base_url=base_url,
            model_name=model_name,
            num_faces=2,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
            ):

        model_path = self.set_model(base_url, model_folder_path, model_name)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=output_face_blendshapes,
            output_facial_transformation_matrixes=output_facial_transformation_matrixes,
            running_mode=mp.tasks.vision.RunningMode.VIDEO
        )
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self.num_landmarks = self.NUM_LMK

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
        self.size = image.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        self.results = self.detector.detect_for_video(mp_image, int(time.time() * 1000))
        self.num_detected_faces = len(self.results.face_landmarks)
        return self.results

    def get_normalized_landmark(self, id_face, id_landmark):
        if self.num_detected_faces == 0:
            print('no face')
            return None
        x = self.results.face_landmarks[id_face][id_landmark].x
        y = self.results.face_landmarks[id_face][id_landmark].y
        z = self.results.face_landmarks[id_face][id_landmark].z
        return np.array([x, y, z])

    def get_landmark(self, id_face, id_landmark):
        if self.num_detected_faces == 0:
            print('no face')
            return None
        height, width = self.size[:2]
        x = self.results.face_landmarks[id_face][id_landmark].x
        y = self.results.face_landmarks[id_face][id_landmark].y
        z = self.results.face_landmarks[id_face][id_landmark].z
        return np.array([int(x*width), int(y*height), int(z*width)], dtype=int)

    def get_landmark_presence(self, id_hand, id_landmark):
        return self.results.face_landmarks[id_hand][id_landmark].presence

    def get_landmark_visibility(self, id_hand, id_landmark):
        return self.results.face_landmarks[id_hand][id_landmark].visibility

    def visualize(self, img):
        annotated_image = np.copy(img)
        for i, face in enumerate(self.results.face_landmarks):
            for j in range(len(face)):
                point = self.get_landmark(i, j)
                cv2.circle(annotated_image, tuple(point[:2]), self.RADIUS_SIZE, self.FONT_COLOR, thickness=self.FONT_THICKNESS)
        return annotated_image

    def visualize_with_mp(self, rgb_image):
        face_landmarks_list = self.results.face_landmarks
        annotated_image = np.copy(rgb_image)
        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]
            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks])
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_iris_connections_style())
        return annotated_image

    def release(self):
        self.detector.close()


def main():
    cap = cv2.VideoCapture(0)
    Face = MediapipeFaceLandmark()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            print("Ignoring empty camera frame.")
            break

        # FaceLandmarker requires horizontal flip for input image
        flipped_frame = cv2.flip(frame, 1)

        results = Face.detect(flipped_frame)

        if Face.num_detected_faces > 0:
            index_face = 0 #
            index_landmark = 0 #
            print(
                Face.get_normalized_landmark(index_face, index_landmark),
                Face.get_landmark(index_face, index_landmark)
                )

        annotated_image = Face.visualize(flipped_frame)
        # annotated_image = Face.visualize_with_mp(flipped_frame)

        cv2.imshow('annotated image', annotated_image)
        key = cv2.waitKey(1)&0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    Face.release()
    cap.release()

if __name__=='__main__':
    main()

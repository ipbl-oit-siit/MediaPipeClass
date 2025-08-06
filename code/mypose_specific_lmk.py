import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import time
from MediapipePoseLandmark import MediapipePoseLandmark as PoseLmk

device = 0 # cameera device number

def get_frame_number(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * fps)
    return frame_now

def draw_pose_landmarks(image, Pose):
    # Draw the all landmarks on the image.
    for id_pose in range(Pose.num_detected_poses): # all poses
        for id_lmk in range(Pose.num_landmarks): # all landmarks
            landmark_point = Pose.get_landmark(id_pose, id_lmk) # get landmark
            cv2.circle(image, landmark_point[:2], 1, (0, 255, 0), 2) # draw landmark

def draw_pose_landmarks_only_left_shoulder(image, Pose):
    # Draw only a TIP landamrk of index finger on the image.
    for id_pose in range(Pose.num_detected_poses): # all poses
        id_lmk = Pose.LEFT_SHOULDER # 11
        landmark_point = Pose.get_landmark(id_pose, id_lmk) # get landmark
        cv2.circle(image, tuple(landmark_point[:2]), 2, (0, 0, 255), 2) # draw landmark
        # write text on the image
        txt = '({:d}, {:d})'.format(landmark_point[0], landmark_point[1])
        tip_point_for_text = (landmark_point[0]-20, landmark_point[1]-20)
        cv2.putText(image, org=tip_point_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_4)

def draw_pose_landmarks_only_basic_points(image, Pose):
    # Draw only TIP landmarks on the image.
    id_list = [0, 11, 12, 15, 16] # nose, left shoulder, right shoulder, left wrist, right wrist
    for id_pose in range(Pose.num_detected_poses): # all poses
        for id_lmk in id_list:
            landmark_point = Pose.get_landmark(id_pose, id_lmk) # get landmark
            cv2.circle(image, tuple(landmark_point[:2]), 2, (0, 255, 0), 2) # draw landmark

def main():
    # For webcam input:
    global device

    cap = cv2.VideoCapture(device)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Size:", ht, "x", wt, "/Fps: ", fps)

    start = time.perf_counter()
    frame_prv = -1

    wname = 'MediaPipe PoseLandmark'
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)

    # make instance of our mediapipe class
    # you can set options
    Pose = PoseLmk()

    while cap.isOpened():
        frame_now = get_frame_number(start, fps)
        if frame_now == frame_prv:
            continue
        frame_prv = frame_now

        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # not flip at PoseLandmark
        results = Pose.detect(frame)

        # [1] Draw the all landmarks on the image.
        # for id_pose in range(Pose.num_detected_poses): # all poses
        #     for id_lmk in range(Pose.num_landmarks): # all landmarks
        #         landmark_point = Pose.get_landmark(id_pose, id_lmk)
        #         cv2.circle(frame, landmark_point[:2], 2, (0, 255, 0), 2)

        # draw_pose_landmarks(frame, Pose)

        draw_pose_landmarks_only_basic_points(frame, Pose)

        draw_pose_landmarks_only_left_shoulder(frame, Pose)

        cv2.imshow(wname, frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    Pose.release()
    cap.release()

if __name__ == '__main__':
    main()
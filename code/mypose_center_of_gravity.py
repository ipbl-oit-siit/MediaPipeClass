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

def draw_pose_landmarks_only_basic_points(image, Pose):
    # Draw only TIP landmarks on the image.
    id_list = [0, 11, 12, 15, 16] # nose, left shoulder, right shoulder, left wrist, right wrist
    for id_pose in range(Pose.num_detected_poses): # all poses
        for id_lmk in id_list:
            landmark_point = Pose.get_landmark(id_pose, id_lmk) # get landmark
            cv2.circle(image, tuple(landmark_point[:2]), 2, (0, 255, 0), 2) # draw landmark

def draw_cog_point_of_basic_points(image, Pose):
    for id_pose in range(Pose.num_detected_poses): # all poses
        pt_cog = np.zeros((3,), dtype=int) # make initialized array: np.array([0, 0, 0])
        id_list = [0, 11, 12, 15, 16] # nose, left shoulder, right shoulder, left wrist, right wrist
        for id_lmk in id_list:
            pt_cog += Pose.get_landmark(id_pose, id_lmk)
        pt_cog = (pt_cog/len(id_list)).astype(int)
        cv2.circle(image, pt_cog[:2], 5, (0, 0, 255), 2) # draw landmark

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

        draw_pose_landmarks_only_basic_points(frame, Pose)

        draw_cog_point_of_basic_points(frame, Pose)

        cv2.imshow(wname, frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    Pose.release()
    cap.release()

if __name__ == '__main__':
    main()
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

def draw_judge_O_X_with_pose(image, Pose):
    for id_pose in range(Pose.num_detected_poses): # all poses
        # keypoints of arms
        pt_left_shoulder = Pose.get_landmark(id_pose, Pose.LEFT_SHOULDER) # 11
        pt_right_shoulder = Pose.get_landmark(id_pose, Pose.RIGHT_SHOULDER) # 12
        pt_left_elbow = Pose.get_landmark(id_pose, Pose.LEFT_ELBOW) # 13
        pt_right_elbow = Pose.get_landmark(id_pose, Pose.RIGHT_ELBOW) # 14
        pt_left_wrist = Pose.get_landmark(id_pose, Pose.LEFT_WRIST) # 15
        pt_right_wrist = Pose.get_landmark(id_pose, Pose.RIGHT_WRIST) # 16

        txt = ''
        if (pt_right_elbow[1] > pt_left_wrist[1]
            and pt_right_elbow[0] < pt_left_elbow[0]
            and pt_right_elbow[0] < pt_right_wrist[0]
            and pt_left_elbow[1] > pt_right_wrist[1]
            and pt_left_wrist[0] < pt_right_wrist[0]
            ):
            txt = 'X'
        elif pt_right_shoulder[1] > pt_right_elbow[1] > pt_right_wrist[1]:
            if pt_left_shoulder[1] > pt_left_elbow[1] > pt_left_wrist[1]:
                if pt_right_shoulder[0] < pt_right_wrist[0] < pt_left_wrist[0] < pt_left_shoulder[0]:
                    txt = 'O'
        pt_nose = Pose.get_landmark(id_pose, Pose.NOSE) # 0
        cv2.putText(image, org=pt_nose[:2], text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2, lineType=cv2.LINE_4)

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

        draw_judge_O_X_with_pose(frame, Pose)

        cv2.imshow(wname, frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    Pose.release()
    cap.release()

if __name__ == '__main__':
    main()
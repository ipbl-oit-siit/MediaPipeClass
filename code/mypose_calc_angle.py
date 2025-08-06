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

def calc_angle(v1, v2):
    v1_n = np.linalg.norm(v1)
    v2_n = np.linalg.norm(v2)
    cos_theta = np.inner(v1, v2) / (v1_n * v2_n)
    return np.rad2deg(np.arccos(cos_theta))

def draw_cross_angle_2arms(image, Pose):
    for id_pose in range(Pose.num_detected_poses):
        # pickup landmark points of wrist and elbow
        pt_le = Pose.get_landmark(id_pose, Pose.LEFT_ELBOW)  # 13
        pt_re = Pose.get_landmark(id_pose, Pose.RIGHT_ELBOW) # 14
        pt_lw = Pose.get_landmark(id_pose, Pose.LEFT_WRIST)  # 15
        pt_rw = Pose.get_landmark(id_pose, Pose.RIGHT_WRIST) # 16

        # draw arm line (elbow - wrist)
        for pt in [pt_le, pt_re, pt_lw, pt_rw]:
            cv2.circle(image, pt[:2], 5, (0, 0, 255), 3)
        cv2.line(image, pt_le[:2], pt_lw[:2], (0, 255, 0))
        cv2.line(image, pt_re[:2], pt_rw[:2], (0, 255, 0))

        vec1 = pt_lw - pt_le # vector (left elbow -> left wrist)
        vec2 = pt_rw - pt_re # vector (right elbow -> left wrist)
        txt = '{:d}'.format(int(calc_angle(vec1, vec2)))
        pt_for_text = (pt_lw[0]+10, pt_lw[1])
        cv2.putText(image, org=pt_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_4)

def draw_right_arm_angle_against_Y_axis(image, Pose):
    for id_pose in range(Pose.num_detected_poses):
        pt_re = Pose.get_landmark(id_pose, Pose.RIGHT_ELBOW) # 14
        pt_rw = Pose.get_landmark(id_pose, Pose.RIGHT_WRIST) # 16

        # draw right arm (right wrist - right elbow)
        cv2.circle(image, pt_re[:2], 5, (0, 0, 255), 3)
        cv2.circle(image, pt_rw[:2], 5, (0, 0, 255), 3)
        cv2.line(image, pt_re[:2], pt_rw[:2], (0, 255, 0))

        vec1 = pt_rw - pt_re # 3d vector (right wrist -> right elbow)
        vec2 = (0, -1) # 2d vector (vertical upward direction)
        angle = calc_angle(vec1[:2], vec2) # vec1 has 3-dimension
        if pt_rw[0] - pt_re[0] < 0:
            angle = 360 - angle
        txt = '{:d}'.format(int(angle))
        pt_for_text = (pt_rw[0]+10, pt_re[1])
        cv2.putText(image, org=pt_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_4)

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

        draw_cross_angle_2arms(frame, Pose)

        # draw_right_arm_angle_against_Y_axis(frame, Pose)

        cv2.imshow(wname, frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    Pose.release()
    cap.release()

if __name__ == '__main__':
    main()
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import time
from MediapipeHandLandmark import MediapipeHandLandmark as HandLmk

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

def draw_open_bend_indexfinger(image, Hand):
    for id_hand in range(Hand.num_detected_hands):
        # pickup landmark points of index finger
        pt_ifmcp = Hand.get_landmark(id_hand, Hand.INDEX_FINGER_MCP)
        pt_ifpip = Hand.get_landmark(id_hand, Hand.INDEX_FINGER_PIP)
        pt_ifdip = Hand.get_landmark(id_hand, Hand.INDEX_FINGER_DIP)

        # draw index finger (MCP - PIP - DIP)
        cv2.circle(image, pt_ifmcp[:2], 5, (0, 0, 255), 3)
        cv2.circle(image, pt_ifpip[:2], 5, (0, 0, 255), 3)
        cv2.circle(image, pt_ifdip[:2], 5, (0, 0, 255), 3)
        cv2.line(image, pt_ifmcp[:2], pt_ifpip[:2], (0, 255, 0))
        cv2.line(image, pt_ifpip[:2], pt_ifdip[:2], (0, 255, 0))

        vec1 = pt_ifmcp - pt_ifpip # vector (pip -> mcp)
        vec2 = pt_ifdip - pt_ifpip # vector (pip -> dip)
        if calc_angle(vec1, vec2) > 140:
            txt = 'open'
        else:
            txt = 'bend'
        pt_for_text = (pt_ifmcp[0]+10, pt_ifmcp[1])
        cv2.putText(image, org=pt_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_4)

def draw_finger_angle_against_Y_axis(image, Hand):
    for id_hand in range(Hand.num_detected_hands):
        pt_iftip = Hand.get_landmark(id_hand, Hand.INDEX_FINGER_TIP)
        pt_ifpip = Hand.get_landmark(id_hand, Hand.INDEX_FINGER_PIP)

        # draw index finger (PIP - TIP)
        cv2.circle(image, pt_ifpip[:2], 5, (0, 0, 255), 3)
        cv2.circle(image, pt_iftip[:2], 5, (0, 0, 255), 3)
        cv2.line(image, pt_ifpip[:2], pt_iftip[:2], (0, 255, 0))

        vec1 = pt_iftip - pt_ifpip # 3d vector (tip -> pip)
        vec2 = (0, -1) # 2d vector (vertical upward direction)
        angle = calc_angle(vec1[:2], vec2) # vec1 has 3-dimension
        if pt_iftip[0] - pt_ifpip[0] < 0:
            angle = 360 - angle
        txt = '{:d}'.format(int(angle))
        pt_for_text = (pt_iftip[0]+10, pt_iftip[1])
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

    wname = 'MediaPipe HandLandmark'
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)

    # make instance of our mediapipe class
    # you can set options
    Hand = HandLmk()

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

        results = Hand.detect(frame)

        draw_open_bend_indexfinger(frame, Hand)

        # draw_finger_angle_against_Y_axis(frame, Hand)

        cv2.imshow(wname, frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    Hand.release()
    cap.release()

if __name__ == '__main__':
    main()
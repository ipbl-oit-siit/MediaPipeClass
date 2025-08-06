import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import time
from MediapipeFaceLandmark import MediapipeFaceLandmark as FaceLmk

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

def draw_nose_angle_against_Y_axis(image, Face):
    for id_face in range(Face.num_detected_faces):
        pt_tn = Face.get_landmark(id_face, 1)    # top of nose
        pt_be = Face.get_landmark(id_face, 6) # between the eyebrows

        # draw nose line
        cv2.circle(image, pt_be[:2], 5, (0, 0, 255), 3)
        cv2.circle(image, pt_tn[:2], 5, (0, 0, 255), 3)
        cv2.line(image, pt_be[:2], pt_tn[:2], (0, 255, 0))

        vec1 = pt_be - pt_tn # 3d vector (top of nose -> between the eyebrows)
        vec2 = (0, -1) # 2d vector (vertical upward direction)
        angle = calc_angle(vec1[:2], vec2) # vec1 has 3-dimension
        if pt_be[0] - pt_tn[0] < 0:
            angle = 360 - angle
        txt = '{:d}'.format(int(angle))
        pt_for_text = (pt_tn[0]+10, pt_tn[1])
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

    wname = 'MediaPipe FaceLandmark'
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)

    # make instance of our mediapipe class
    # you can set options
    Face = FaceLmk()

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

        # Flip the image horizontally
        flipped_frame = cv2.flip(frame, 1) ### very important ####

        results = Face.detect(flipped_frame)

        draw_nose_angle_against_Y_axis(flipped_frame, Face)

        cv2.imshow(wname, flipped_frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    Face.release()
    cap.release()

if __name__ == '__main__':
    main()
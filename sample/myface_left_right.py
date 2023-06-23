import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import time
from MediapipeFaceLandmark import MediapipeFaceLandmark as FaceLmk

device = 0 # cameera device number

def get_frame_number(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * 1000 / fps)
    return frame_now

def draw_left_right_with_face(image, Face):
    for id_face in range(Face.num_detected_faces):
        # facial keypoints
        pt_top = Face.get_landmark(id_face, 10)
        pt_bottom = Face.get_landmark(id_face, 152)
        pt_left = Face.get_landmark(id_face, 234)
        pt_right = Face.get_landmark(id_face, 454)
        pt_center = Face.get_landmark(id_face, 0)

        # center of gravity
        pt_cog = np.zeros((3,), dtype=int)
        for id_lmk in range(Face.num_landmarks):
            pt_cog += Face.get_landmark(id_face, id_lmk)
        pt_cog = (pt_cog/Face.num_landmarks).astype(int)

        l = pt_cog[0] - pt_left[0]
        r = pt_right[0] - pt_cog[0]

        if abs(l) > 5*abs(r):
            txt = 'right'
        elif 5*abs(l) < abs(r):
            txt = 'left'
        else:
            txt = ''
        pt_for_text = (pt_top[0]+10, pt_top[1])
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

        draw_left_right_with_face(flipped_frame, Face)

        cv2.imshow(wname, flipped_frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    Face.release()
    cap.release()

if __name__ == '__main__':
    main()
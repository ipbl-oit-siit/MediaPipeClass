import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import time
from MediapipeFaceDetection import MediapipeFaceDetection as FaceDetect

device = 0 # cameera device number

def get_frame_number(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * fps)
    return frame_now

def draw_face_keypoints_boundingbox(image, FaceDtc):
    for id_face in range(FaceDtc.num_detected_faces):
        bx, by, bw, bh = FaceDtc.get_bounding_box(id_face)
        cv2.rectangle(image, (bx, by), (bx+bw, by+bh), (0,255,0), 2)
        for id_keypoint in range(FaceDtc.num_landmarks):
            keypoint = FaceDtc.get_landmark(id_face, id_keypoint)
            cv2.circle(image, tuple(keypoint), 2, (0, 0, 255), 3)

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
    FaceDtc = FaceDetect()

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

        results = FaceDtc.detect(flipped_frame)

        draw_face_keypoints_boundingbox(flipped_frame, FaceDtc)

        cv2.imshow(wname, flipped_frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    FaceDtc.release()
    cap.release()

if __name__ == '__main__':
    main()
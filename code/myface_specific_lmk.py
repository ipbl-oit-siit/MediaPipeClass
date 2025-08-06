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

def draw_face_landmarks(image, Face):
    # Draw the all landmarks on the image.
    for id_face in range(Face.num_detected_faces): # all faces
        for id_lmk in range(Face.num_landmarks): # all landmarks
            landmark_point = Face.get_landmark(id_face, id_lmk) # get landmark
            cv2.circle(image, landmark_point[:2], 2, (0, 255, 0), 2) # draw landmark

def draw_face_landmarks_only_left_point(image, Face):
    # Draw only a TIP landamrk of index finger on the image.
    for id_face in range(Face.num_detected_faces): # all faces
        id_lmk = 234 # left
        landmark_point = Face.get_landmark(id_face, id_lmk) # get landmark
        cv2.circle(image, tuple(landmark_point[:2]), 2, (0, 0, 255), 2) # draw landmark
        # write text on the image
        txt = '({:d}, {:d})'.format(landmark_point[0], landmark_point[1])
        tip_point_for_text = (landmark_point[0]-20, landmark_point[1]-20)
        cv2.putText(image, org=tip_point_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_4)

def draw_face_landmarks_only_basic_points(image, Face):
    # Draw only TIP landmarks on the image.
    id_list = [0, 10, 152, 234, 454] # center, top, bottom, left, right
    for id_face in range(Face.num_detected_faces): # all faces
        for id_lmk in id_list:
            landmark_point = Face.get_landmark(id_face, id_lmk) # get landmark
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

        # [1] Draw the all landmarks on the image.
        # for id_face in range(Face.num_detected_faces): # all faces
        #     for id_lmk in range(Face.num_landmarks): # all landmarks
        #         landmark_point = Face.get_landmark(id_face, id_lmk)
        #         cv2.circle(flipped_frame, landmark_point[:2], 2, (0, 255, 0), 2)

        # draw_face_landmarks(flipped_frame, Face)

        draw_face_landmarks_only_basic_points(flipped_frame, Face)

        draw_face_landmarks_only_left_point(flipped_frame, Face)

        cv2.imshow(wname, flipped_frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    Face.release()
    cap.release()

if __name__ == '__main__':
    main()
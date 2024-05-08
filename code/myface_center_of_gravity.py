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

def draw_face_landmarks_only_basic_points(image, Face):
    # Draw only TIP landmarks on the image.
    id_list_tip = [0, 10, 152, 234, 454] # center, top, bottom, left, right
    for id_face in range(Face.num_detected_faces): # all faces
        for id_lmk in id_list_tip:
            landmark_point = Face.get_landmark(id_face, id_lmk) # get landmark
            cv2.circle(image, tuple(landmark_point[:2]), 2, (0, 255, 0), 2) # draw landmark

def draw_cog_point_of_all_tips(image, Face):
    for id_face in range(Face.num_detected_faces): # all faces
        pt_cog = np.zeros((3,), dtype=int) # make initialized array: np.array([0, 0, 0])
        id_list = [0, 10, 152, 234, 454] # center, top, bottom, left, right
        for id_lmk in id_list:
            pt_cog += Face.get_landmark(id_face, id_lmk)
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

        draw_face_landmarks_only_basic_points(flipped_frame, Face)

        draw_cog_point_of_all_tips(flipped_frame, Face)

        cv2.imshow(wname, flipped_frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    Face.release()
    cap.release()

if __name__ == '__main__':
    main()
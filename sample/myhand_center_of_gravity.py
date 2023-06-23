import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import time
from MediapipeHandLandmark import MediapipeHandLandmark as HandLmk

device = 0 # cameera device number

def get_frame_number(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * 1000 / fps)
    return frame_now

def draw_hand_landmarks_only_tip(image, Hand):
    # Draw only TIP landmarks on the image.
    id_list_tip = [4, 8, 12, 16, 20]
    for id_hand in range(Hand.num_detected_hands): # all hands
        for id_lmk in id_list_tip: # only TIP landmarks
            landmark_point = Hand.get_landmark(id_hand, id_lmk) # get landmark
            cv2.circle(image, tuple(landmark_point[:2]), 2, (0, 255, 0), 2) # draw landmark

def draw_cog_point_of_all_tips(image, Hand):
    for id_hand in range(Hand.num_detected_hands): # all hands
        pt_cog = np.zeros((3,), dtype=int) # make initialized array: np.array([0, 0, 0])
        id_list_tip = [4, 8, 12, 16, 20]
        for id_lmk in id_list_tip:
            pt_cog += Hand.get_landmark(id_hand, id_lmk)
        pt_cog = (pt_cog/len(id_list_tip)).astype(int)
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

        # Flip the image horizontally
        flipped_frame = cv2.flip(frame, 1) ### very important ####

        results = Hand.detect(flipped_frame)

        draw_hand_landmarks_only_tip(flipped_frame, Hand)

        draw_cog_point_of_all_tips(flipped_frame, Hand)

        cv2.imshow(wname, flipped_frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    Hand.release()
    cap.release()

if __name__ == '__main__':
    main()
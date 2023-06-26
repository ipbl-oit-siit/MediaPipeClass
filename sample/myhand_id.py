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

def draw_hand_landmarks_with_id(image, Hand):
    # Draw the all landmarks on the image.
    for id_hand in range(Hand.num_detected_hands): # all hands
        for id_lmk in range(Hand.num_landmarks): # all landmarks
            landmark_point = Hand.get_landmark(id_hand, id_lmk) # get landmark
            cv2.circle(image, landmark_point[:2], 2, (0, 255, 0), 2) # draw landmark

        txt = 'id='+str(id_hand)
        pt_wrist = Hand.get_landmark(id_hand, Hand.WRIST) # Hand.WRIST=0
        wrist_point_for_text = (pt_wrist[0]+30, pt_wrist[1]+10)
        cv2.putText(image, org=wrist_point_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2, lineType=cv2.LINE_4)

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

    cnt = 0
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

        # [1] Draw the all landmarks on the image.
        draw_hand_landmarks_with_id(flipped_frame, Hand)

        cv2.imshow(wname, flipped_frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        if cnt>200:
            input()
            cnt=0
        cnt+=1

    cv2.destroyAllWindows()
    Hand.release()
    cap.release()

if __name__ == '__main__':
    main()
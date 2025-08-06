import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import time
from MediapipeHandLandmark import MediapipeHandLandmark as HandLmk

device = 0 # cameera device number

def get_frame_number(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * fps)
    return frame_now

def draw_hand_landmarks(image, Hand):
    # Draw the all landmarks on the image.
    for id_hand in range(Hand.num_detected_hands): # all hands
        for id_lmk in range(Hand.num_landmarks): # all landmarks
            landmark_point = Hand.get_landmark(id_hand, id_lmk) # get landmark
            cv2.circle(image, landmark_point[:2], 2, (0, 255, 0), 2) # draw landmark

def draw_hand_landmarks_only_tip_of_indexfinger(image, Hand):
    # Draw only a TIP landamrk of index finger on the image.
    for id_hand in range(Hand.num_detected_hands): # all hands
        id_lmk = Hand.INDEX_FINGER_TIP # 8
        landmark_point = Hand.get_landmark(id_hand, id_lmk) # get landmark
        cv2.circle(image, tuple(landmark_point[:2]), 2, (0, 0, 255), 2) # draw landmark
        # write text on the image
        txt = '({:d}, {:d})'.format(landmark_point[0], landmark_point[1])
        tip_point_for_text = (landmark_point[0]-20, landmark_point[1]-20)
        cv2.putText(image, org=tip_point_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_4)

def draw_hand_landmarks_only_tip(image, Hand):
    # Draw only TIP landmarks on the image.
    id_list_tip = [4, 8, 12, 16, 20]
    for id_hand in range(Hand.num_detected_hands): # all hands
        for id_lmk in id_list_tip: # only TIP landmarks
            landmark_point = Hand.get_landmark(id_hand, id_lmk) # get landmark
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

        # [1] Draw the all landmarks on the image.
        # for id_hand in range(Hand.num_detected_hands): # all hands
        #     for id_lmk in range(Hand.num_landmarks): # all landmarks
        #         landmark_point = Hand.get_landmark(id_hand, id_lmk)
        #         cv2.circle(frame, landmark_point[:2], 2, (0, 255, 0), 2)

        # draw_hand_landmarks(frame, Hand)

        draw_hand_landmarks_only_tip(frame, Hand)

        draw_hand_landmarks_only_tip_of_indexfinger(frame, Hand)

        cv2.imshow(wname, frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    Hand.release()
    cap.release()

if __name__ == '__main__':
    main()
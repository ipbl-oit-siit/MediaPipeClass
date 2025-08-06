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

def draw_hands_with_handedness(image, Hand):
    RIGHT_HAND_COLOR = (0, 255, 0)
    LEFT_HAND_COLOR = (100, 100, 255)
    H_MARGIN = 10  # pixels
    V_MARGIN = 30  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1

    for id_hand in range(Hand.num_detected_hands):
        handedness = Hand.get_handedness(id_hand)
        score = Hand.get_score_handedness(id_hand)
        wrist_point = Hand.get_landmark(id_hand, 0)

        if handedness == 'Right':
            color = RIGHT_HAND_COLOR
        else:
            color = LEFT_HAND_COLOR

        for id_lmk in range(Hand.num_landmarks):
            landmark_point = Hand.get_landmark(id_hand, id_lmk)
            cv2.circle(image, tuple(landmark_point[:2]), 1, color, 2)

        txt = handedness+'('+'{:#.2f}'.format(score)+')'
        wrist_point_for_text = (wrist_point[0]+H_MARGIN, wrist_point[1]+V_MARGIN)
        cv2.putText(image, org=wrist_point_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=FONT_SIZE, color=color, thickness=FONT_THICKNESS, lineType=cv2.LINE_4)

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

        draw_hands_with_handedness(frame, Hand)

        cv2.imshow(wname, frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    Hand.release()
    cap.release()

if __name__ == '__main__':
    main()
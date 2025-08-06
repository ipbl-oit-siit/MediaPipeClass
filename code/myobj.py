import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import time
from MediapipeObjectDetection import MediapipeObjectDetection as ObjDtc

device = 0 # cameera device number

def get_frame_number(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * fps)
    return frame_now

def visualize_objectname_boundingbox(image, Obj):
    for id_object in range(Obj.num_detected_objects):
        category_name = Obj.get_category_name(id_object)
        category_score = Obj.get_category_score(id_object)
        bx, by, bw, bh = Obj.get_bounding_box(id_object) # x, y, w, h

        pt_upper_left = (bx, by)
        pt_lower_right = (bx + bw, by + bh)
        cv2.rectangle(image, pt_upper_left, pt_lower_right, (0,255,0), 2)

        txt = category_name+'({:#.2f})'.format(category_score)
        pt_for_text = (bx + 10, by + 30)
        cv2.putText(image, org=pt_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_4)

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

    wname = 'MediaPipe ObjectDetection'
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)

    # make instance of our mediapipe class
    # you can set options
    Obj = ObjDtc(score_threshold=0.3)

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

        # not flip at PoseLandmark
        results = Obj.detect(frame)

        visualize_objectname_boundingbox(frame, Obj)

        cv2.imshow(wname, frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    Obj.release()
    cap.release()

if __name__ == '__main__':
    main()
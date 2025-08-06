import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import time
from MediapipeImageSegmentation import MediapipeImageSegmentation as ImgSeg

device = 0 # cameera device number

def get_frame_number(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * fps)
    return frame_now

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

    wname = 'MediaPipe ImageSegmentation'
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)

    # make instance of our mediapipe class
    # you can set options
    Seg = ImgSeg()

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
        results = Seg.detect(frame)

        segmented_masks = Seg.get_segmentation_masks()
        face_skin_mask_binary = (segmented_masks == Seg.FACE_SKIN)
        face_skin_confidence_mask = Seg.get_confidence_mask(Seg.FACE_SKIN)
        masks = cv2.hconcat([face_skin_mask_binary.astype(float), face_skin_confidence_mask.astype(float)])

        # Display normalized mask in pseudo-color
        normalized_masks = Seg.get_normalized_masks()
        cv2.imshow('multiclass mask', cv2.applyColorMap(normalized_masks, cv2.COLORMAP_JET))

        cv2.imshow(wname, masks) ### masks
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    Seg.release()
    cap.release()

if __name__ == '__main__':
    main()
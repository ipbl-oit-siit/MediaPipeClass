import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import time
from MediapipePoseLandmark import MediapipePoseLandmark as PoseLmk

device = 0 # cameera device number

def get_frame_number(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * 1000 / fps)
    return frame_now

def visualize_segmentation_mask(image, Pose, id_pose):
    seg_mask = Pose.get_segmentation_mask(id_pose)
    # all_seg_mask = Pose.get_all_segmentation_masks()

    normalized_seg_mask = seg_mask.astype(float)/np.max(seg_mask) # normalize [0.0, 1.0]
    mask = np.tile(normalized_seg_mask[:,:,None], [1,1,3])*0.7 + 0.3
    return (image * mask).astype(np.uint8)

def visualize_all_segmentation_mask(image, Pose):
    all_seg_mask = Pose.get_all_segmentation_masks()

    normalized_seg_mask = all_seg_mask.astype(float)/np.max(all_seg_mask) # normalize [0.0, 1.0]
    mask = np.tile(normalized_seg_mask[:,:,None], [1,1,3])*0.7 + 0.3
    return (image * mask).astype(np.uint8)

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

    wname = 'MediaPipe PoseLandmark'
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)

    # make instance of our mediapipe class
    # you can set options
    Pose = PoseLmk()

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
        results = Pose.detect(frame)

        annotated_frame = frame.copy()
        if Pose.num_detected_poses > 0:
            # id_pose = 0
            # annotated_frmae = visualize_segmentation_mask(frame, Pose, id_pose)

            annotated_frame = visualize_all_segmentation_mask(frame, Pose)

        cv2.imshow(wname, annotated_frame) #### annotated frame
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    Pose.release()
    cap.release()

if __name__ == '__main__':
    main()
# How to use more

[back to the top page of MEdiaPipe Class](../README.md)


---
## How to use landmarks
- `HandLandmark`, `FaceLandmark`, `PoseLandmark` detect landmarks. 
    - `HandLandmark`: 21 points<br>
    <image src="https://developers.google.com/static/mediapipe/images/solutions/hand-landmarks.png" width="50%" height="50%"><br>
    - `FaceLandmark`: 478 points<br>
    <image src="https://developers.google.com/static/mediapipe/images/solutions/face_landmarker_keypoints.png" width="20%" height="20%">
    <image src="https://user-images.githubusercontent.com/48384506/113843509-dfcd3680-97a8-11eb-8fd5-e16bf48a113b.png" width="25%" height="25%">
    <br>
    - `PoseLandmark`: 33 points<br>
    <image src="https://developers.google.com/static/mediapipe/images/solutions/pose_landmarks_index.png" width="20%" height="20%"><br>
- The following is some examples of how to refer landmarks.
- The following is samples of `hand`, which can be used as is by replacing `Hand` with {`Face`, `Pose`} and `hand` with {`face`, `pose`})

### ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) Abstract
- The detection results of all {hands, faces, poses} are stored in the variable `results`, but its data structure is complex. Therefore, we provide several class variables and getter functions in our mediapipe class.
- After `Hand.detect(frame)`, you can call following class variables and getter functions.
    - `Hand.num_detected_hands`: The number of detected hands (max is `2` in the default setting). If `0`, it's dangerous to continue the process. An error may occur when some referencing.
    - `Hand.num_landmarks`: The number of hand landmarks. Basically, it's `21` in HandLandmark.
    - `landmark_point = Hand.get_landmark(id_hand, id_landmark)`
        - `landmark_point`: The coordinate array of `id_landmark`-th landmark of `id_hand`-th pose. Type is `np.ndarray([x, y, z], dtype=int)`
    - `presence = Hand.get_landmark_presence(id_hand, id_landmark)`: The presence of `id_landmark`-th landmark of `id_hand`-th hand. If low, the validity is low.
    - `visibility = Hand.get_landmark_visibility(id_hand, id_landmark)`: The presence of `id_landmark`-th landmark of `id_hand`{hands, faces, poses}-th hand. If low, the validity is low.
- For other details, please refer to each specification page.
    - [HandLandmark](HandLandmark.md)
    - [HandGestureRecognition](HandGestureRecognition.md)
    - [FaceLandmark](FaceLandmark.md)
    - [PoseLandmark](PoseLandmark.md)

### ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) how to refer the all landmarks
- sample code
```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import time
from MediapipeHandLandmark import MediapipeHandLandmark as HandLmk

device = 0 # cameera device number

def main():
    # For webcam input:
    global device

    cap = cv2.VideoCapture(device)
    Hand = HandLmk()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        flipped_frame = cv2.flip(frame, 1)        

        results = Hand.detect(flipped_frame)

        # Draw the all landmarks on the image.
        for id_hand in range(Hand.num_detected_hands): # all hands
            for id_lmk in range(Hand.num_landmarks): # all landmarks
                landmark_point = Hand.get_landmark(id_hand, id_lmk)
                cv2.circle(flipped_frame, (landmark_point[0], landmark_point[1]), 1, (0, 0, 255), 2)
                # cv2.circle(flipped_frame, tuple(landmark_point[:2]), 1, (0, 0, 255), 2)

        cv2.imshow('MediaPipe HandLandmark', flipped_frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
main()
```
- function sample
```python
def draw_hand_landmarks(image, Hand):
    # Draw the all landmarks on the image.
    for id_hand in range(Hand.num_detected_hands): # all hands
        for id_lmk in range(Hand.num_landmarks): # all landmarks
            landmark_point = Hand.get_landmark(id_hand, id_lmk) # get landmark
            cv2.circle(image, tuple(landmark_point[:2]), 1, (0, 0, 255), 2) # draw landmark
```
### ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) how to refer only to specific landmarks
- function sample
```python
def draw_hand_landmarks_only_tip(image, Hand):
    # Draw only TIP landmarks on the image.
    id_list_tip = [4, 8, 12, 16, 20]
    for id_hand in range(Hand.num_detected_hands): # all hands
        for id_lmk in id_list_tip: # only TIP landmarks
            landmark_point = Hand.get_landmark(id_hand, id_lmk) # get landmark
            cv2.circle(image, tuple(landmark_point[:2]), 1, (0, 0, 255), 2) # draw landmark
```
### ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) how to make landmark array
- If you want to use the all landmark data as an list for all your processing, the following sample will help.
- function sample
```python
def make_hand_landmarks_array(image, Hand):
    # Draw the all landmarks on the image.
    hand_landmarks = []
    for id_hand in range(Hand.num_detected_hands): # all hands
        landmark_point = []
        for id_lmk in range(Hand.num_landmarks): # all landmarks
            landmark_point.append(Hand.get_landmark(id_hand, id_lmk))
        hand_landmarks.append(landmark_point)
    return hand_landmarks
```
- how to use
```python
# list ----------------------------------------------
hand_landmarks = make_hand_landmarks_array(flipped_frame, Hand)
point1 = hand_landmarks[0][1] # 1-th landmark of 0-th hand
point2 = hand_landmarks[1][1] # 1-th landmark of 1-th hand
vec = point1 - point2 # vector (point2 -> point1)

for hand in hand_landmarks: # all hands
    for landmark in hand: # all landmarks
        cv2.circle(flipped_frame, tuple(landmark[:2]), 1, (0, 0, 255), 2)

id_list_tip = [4, 8, 12, 16, 20]
for hand in hand_landmarks: # all hands
    for index, landmark in enumerate(hand): # all landmarks
        if index in id_list_tip: # only TIP landmarks
            cv2.circle(flipped_frame, tuple(landmark[:2]), 1, (0, 0, 255), 2)
```
```python
# getter --------------------------------------------
point1 = Hand.get_landmark(0, 1)
point2 = Hand.get_landmark(1, 1)
vec = point1 - point2 # vector (point2 -> point1)

for id_hand in range(Hand.num_detected_hands): # all hands
    for id_lmk in range(Hand.num_landmarks): # all landmarks
        landmark_point = Hand.get_landmark(id_hand, id_lmk) # get landmark
        cv2.circle(image, tuple(landmark_point[:2]), 1, (0, 0, 255), 2) # draw landmark

id_list_tip = [4, 8, 12, 16, 20]
for id_hand in range(Hand.num_detected_hands): # all hands
    for id_lmk in id_list_tip: # only TIP landmarks
        landmark_point = Hand.get_landmark(id_hand, id_lmk) # get landmark
        cv2.circle(image, tuple(landmark_point[:2]), 1, (0, 0, 255), 2) # draw landmark
```

### ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) how to calculate the angle between 2 vectors
- function `calc_angle`: calculate the angle between 2 vectors
```python
def calc_angle(v1, v2):
    v1_n = np.linalg.norm(v1)
    v2_n = np.linalg.norm(v2)
    cos_theta = np.inner(v1, v2) / (v1_n * v2_n)
    return np.rad2deg(np.arccos(cos_theta))
```
- :o:[Sample 1] how to check the open/close of the index finger
```python
id_hand = 0
pt_mcp = Hand.get_landmark(id_hand, Hand.INDEX_FINGER_MCP)
pt_pip = Hand.get_landmark(id_hand, Hand.INDEX_FINGER_PIP)
pt_dip = Hand.get_landmark(id_hand, Hand.INDEX_FINGER_DIP)
vec1 = pt_mcp - pt_pip # vector (pip -> mcp)
vec2 = pt_dip - pt_pip # vector (pip -> dip)
if calc_angle(vec1, vec2) > 140:
    print('The index finger is open.')
else:
    print('The index finger is close.')
```
- :o:[Sample 2] how to check the 2d-angle between the vertical upward direction and the direction pointed by the index finger
```python
id_hand = 0
pt_tip = Hand.get_landmark(id_hand, Hand.INDEX_FINGER_TIP)
pt_pip = Hand.get_landmark(id_hand, Hand.INDEX_FINGER_PIP)
vec1 = pt_mcp - pt_pip # vector (pip -> mcp)
vec2 = (0, -1)

angle = calc_angle(vec1[:2], vec2) # vec1 has z-coordinate
if pt_tip[0] - pt_pip[0] < 0:
    angle = 360 - angle
print(angle)
```

---
## Original contents of each class
### ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) `HandLandmark`
- `HandLandmark` can discriminate between left and right hand.
- The following is sample function to draw with different colors for the left and right hands.
#### :o:[Sample] draw with different colors for the left and right hands
- sample function
```python
def draw_hands_with_handedness(image, Hand):
    RIGHT_HAND_COLOR = (0, 255, 0)
    LEFT_HAND_COLOR = (100, 100, 255)

    for id_hand in range(Hand.num_detected_hands):
        handedness = Hand.get_handedness(id_hand)
        score = Hand.get_score_handedness(id_hand)
        wrist_point = Hand.get_landmark(i, 0)

        if handedness == 'Right':
            color = RIGHT_HAND_COLOR
        else:
            color = LEFT_HAND_COLOR

        for id_lmk in range(Hand.num_landmarks):
            landmark_point = Hand.get_landmark(id_hand, id_lmk)
            cv2.circle(image, tuple(landmark_point[:2]), 1, color, 2)

        txt = handedness+'('+'{:#.2f}'.format(score)+')'
        wrist_point_for_text = (wrist_point[0]+self.H_MARGIN, wrist_point[1]+self.V_MARGIN)
        cv2.putText(image, org=wrist_point_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=self.FONT_SIZE, color=color, thickness=self.FONT_THICKNESS, lineType=cv2.LINE_4)
```

### ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) `FaceLandmark`
- The following is sample to determine `left` or `right` according to the orientation of the face.
- There are several judgment methods, but the simple one is a judgment method that compares the x-coordinates of several landmarks.<br>
    <image src="../image/q2_face.gif" width="30%" height="30%"><br>
#### :o:[Sample] determine `left` or `right` orientation of the face
- sample function
```python
def judge_left_right_with_face(Face, id_face):
    # facial keypoints
    pt_top = Face.get_landmark(id_face, 10)
    pt_bottom = Face.get_landmark(id_face, 152)
    pt_left = Face.get_landmark(id_face, 234)
    pt_right = Face.get_landmark(id_face, 454)
    pt_center = Face.get_landmark(id_face, 0)

    # center of gravity
    pt_cog = np.zeros((3,), dtype=int)
    for id_lmk in range(Face.num_landmarks):
        pt_cog += Face.get_landmark(id_face, id_lmk)
    pt_cog = (pt_cog/Face.num_landmarks).astype(int)

    l = pt_cog[0] - pt_left[0]
    r = pt_right[0] - pt_cog[0]

    if abs(l) > 5*abs(r):
        return 'right'
    elif 5*abs(l) < abs(r):
        return 'left'
    else:
        return ''
```

### ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) `FaceDetection`
- `FaceDetection` detects all faces in the input image and returns the 2d coordinates of the main key points.
    - keypoints
        - `LEFT_EYE = 0`
        - `RIGHT_EYE = 1`
        - `NOSE_TIP = 2`
        - `MOUTH = 3`
        - `LEFT_EYE_TRAGION = 4`
        - `RIGHT_EYE_TRAGION = 5`
#### :o:[Sample] show all keypoint and bounding box
- sample function
```python
def draw_face_keypoints_boundingbox(image, FaceDect)
    for id_face in range(FaceDect.num_detected_faces):
        bx, by, bw, bh = FaceDect.get_bounding_box(id_face)
        cv2.rectangle(image, (bx, by), (bx+bw, by+bh), (255,0,0), 3)
        for id_keypoint in range(FaceDect.num_landmarks):
            keypoint = FaceDect.get_landmark(id_face, id_keypoint)
            cv2.circle(image, tuple(keypoint), thickness, color, radius)
```

### ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) `PoseLandmark`
- `PoseLandmark` frequently produces landmarks of poor quality due to the difficulty of fitting the body into the camera's angle of view. You should be aware of that in your application development.
- `PoseLandmark` also outputs a segmentation mask for each detected person.
#### :o:[Sample 1] calculate center of gravity of only visible landmarks
- sample function
```python
def calc_cog_only_visible(Pose, id_face):
    # center of gravity
    pt_cog_visible = np.zeros((3,), dtype=int)
    cnt = 0
    for id_lmk in range(Pose.num_landmarks):
        if Pose.get_landmark_visibility(id_pose, id_lmk) > 0.5: # only high visibility landmarks
            pt_cog_visible += Pose.get_landmark(id_pose, id_lmk)
            cnt += 1
    return (pt_cog_visible/cnt).astype(int)
```
- you can also use presence score.
```python
if Pose.get_landmark_visibility(id_pose, id_lmk) > 0.5 and Pose.get_landmark_presence(id_pose, id_lmk) > 0.5:
```
#### :o:[Sample 2] determine which hand is up
- Display "right" when you raise your right hand and "left" when you raise your left hand. In addition, display "both" when you raise your both hands.<br>
    <image src="../image/pose_q2-1.png" width="30%" height="30%"> <image src="../image/pose_q2-2.png" width="30%" height="30%"> <image src="../image/pose_q2-3.png" width="30%" height="30%"><br>
- sample function
```python
def judge_hand_up(Pose, id_pose):
    # keypoints
    pt_nose = Pose.get_landmark(id_pose, Pose.NOSE) # 0
    pt_left_index = Pose.get_landmark(id_pose, Pose.LEFT_INDEX) # 19
    pt_right_index = Pose.get_landmark(id_pose, Pose.RIGHT_INDEX) # 20

    if pt_nose[1] > pt_right_index[1] and pt_nose[1] > pt_left_index[1]:
        return 'both'
    elif pt_nose[1] > pt_right_index[1]:
        return 'left'
    elif pt_nose[1] > pt_left_index[1]:
        return 'right'
    else:
        return ''
```
#### :o:[Sample 3] determine `O` or `X` of arm shape
- Make a shape of "O" or "X" with your arm and display it on the screen according to the shape.
- It is possible to judge by comparing the x-coordinate and y-coordinate values between multiple landmarks.<br>
    <image src="../image/pose_q3.gif" width="30%" height="30%">
- sample code
```python
def judge_O_X_with_pose(Pose, id_pose):
    # keypoints of arms
    pt_left_shoulder = Pose.get_landmark(id_pose, Pose.LEFT_SHOLDER) # 11
    pt_right_shoulder = Pose.get_landmark(id_pose, Pose.RIGHT_SHOLDER) # 12
    pt_left_elbow = Pose.get_landmark(id_pose, Pose.LEFT_ELBOW) # 13
    pt_right_elbow = Pose.get_landmark(id_pose, Pose.RIGHT_ELBOW) # 14
    pt_left_wrist = Pose.get_landmark(id_pose, Pose.LEFT_WRIST) # 15
    pt_right_wrist = Pose.get_landmark(id_pose, Pose.RIGHT_WRIST) # 16

    if (pt_right_elbow[1] > pt_left_wrist[1]
        and pt_right_elbow[0] < pt_left_elbow[0]
        and pt_right_elbow[0] < pt_right_wrist[0]
        and pt_left_elbow[1] > pt_right_wrist[1]
        and pt_left_wrist[0] < pt_right_wrist[0]
        ):
        return 'X'
    elif pt_right_shoulder[1] > pt_right_elbow[1] > pt_right_wrist[1]:
        if pt_left_shoulder[1] > pt_left_elbow[1] > pt_left_wrist[1]:
            if pt_right_shoulder[0] < pt_right_wrist[0] < pt_left_wrist[0] < pt_left_shoulder[0]:
                return 'O'
```
#### :o:[Sample 4] show segmentation mask
- The following sample code reduces the brightness outside of the detected person area.
- The pixel value of the segmentation mask represents the confidence score of personhood.
- sample code
```python
def visualize_segmentation_mask(image, Pose, id_pose):
    seg_mask = Pose.get_segmentation_mask(id_pose)
    # all_seg_mask = Pose.get_all_segmentation_masks()

    normalized_seg_mask = seg_mask.astype(float)/np.max(seg_mask) # normalize [0.0, 1.0]
    mask = np.tile(normalized_seg_mask[:,:,None], [1,1,3])*0.7 + 0.3
    return (image * mask).astype(np.uint8)
```

### ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) `ImageSegmentation`
- `ImageSegmentation` returns the segmentation mask. Its pixel value represents the corresponding segmentation ID.
- In the `selfie_multiclass_256x256.tflite` model, it is as follows.
    - `BACKGROUND = 0`
    - `HAIR = 1`
    - `BODY_SKIN = 2`
    - `FACE_SKIN = 3`
    - `CLOTHES = 4`
    - `OTHERS = 5`
#### :o:[Sample] show segmentation mask for selfie
- sample code
    - `ImgSeg`: instance of MediapipeImageSegmentation Class
```python
segmented_masks = ImgSeg.detect(frame)
# face skin pixels have 'True', others have 'False')
face_skin_mask_binary = (segmented_masks == ImgSeg.FACE_SKIN)

# getter (face skin pixels have '255', others have '0')
face_skin_mask = ImgSeg.get_segmentation_mask(ImgSeg.FACE_SKIN)
```
- You can get the confidence for the segmentation mask of each segmentation ID as a mask.
```python
# Range [0.0, 1.0]
face_skin_confidence_mask = ImgSeg.get_confidence_mask(ImgSeg.FACE_SKIN)
```

### ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) `ObjectDetection`
- `ObjectDetection` returns the category name and bounding box of the detected object.
    - [target object list](https://storage.googleapis.com/mediapipe-tasks/object_detector/labelmap.txt)
#### :o:[Sample] show object's name and bounding box
- sample code
```python
def visualize_objectname_boundingbox(image, Obj)
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
```

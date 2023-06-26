# How to use more

[back to the top page of MEdiaPipe Class](../README.md)


---
## :green_square: How to use landmarks
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

### :red_square: Abstract
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

### :red_square: How to refer the all landmarks
#### :white_square_button: on HandLandmark
<image src="../image/myhand.jpg" width="30%" height="30%"><br>
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

            # [1] Draw the all landmarks on the image.
            for id_hand in range(Hand.num_detected_hands): # all hands
                for id_lmk in range(Hand.num_landmarks): # all landmarks
                    landmark_point = Hand.get_landmark(id_hand, id_lmk)
                    cv2.circle(flipped_frame, landmark_point[:2], 1, (0, 255, 0), 2)

            cv2.imshow(wname, flipped_frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        Hand.release()
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
                cv2.circle(image, landmark_point[:2], 1, (0, 255, 0), 2) # draw landmark
    ```
    ```python
    # [1] Draw the all landmarks on the image.
    draw_hand_landmarks(flipped_frame, Hand)
    cv2.imshow(wname, flipped_frame)
    ```
#### :white_square_button: Samples
**:exclamation: Note that these programs must be placed in the same directory as `our MediaPipe Class file` to work.**
- HandLandmark
    - [myhand.py](../sample/myhand.py)
- PoseLandmark
    - [mypose.py](../sample/mypose.py)<br>
    <image src="../image/mypose.jpg" width=20%>
- FaceLandmark
    - [myface.py](../sample/myface.py)<br>
    <image src="../image/myface.jpg" width=20%>

### :red_square: How to assign id (`id_hand`) by MediaPipe
- MediaPipe assigns the same id to each hand in the order in which it is found, until it is lost.
- :exclamation: Note that if a hand with an earlier id is lost when multiple hands are recognized, the id will be shifted.
    - For example, with the right hand (id=0) and the left hand (id=1) recognized, if the right hand is hidden, the id of the left hand becomes 0.<br>
    <image src="../image/myhand_id_1.jpg" width=25%> <image src="../image/myhand_id_2.jpg" width=25%> <image src="../image/myhand_id_3.jpg" width=25%><br>

#### :white_square_button: Samples
**:exclamation: Note that these programs must be placed in the same directory as `our MediaPipe Class file` to work.**
- [myhand_id.py](../sample/myhand_id.py)

### :red_square: How to refer only to specific landmarks
#### :white_square_button: on `HandLandmarker`
- :o: [Sample1] How to show only TIP of index finger<br>
    <image src="../image/myhand_tip_indexfinger.jpg" width=25%>
    ```python
    def draw_hand_landmarks_only_tip_of_indexfinger(image, Hand):
        # Draw only a TIP landamrk of index finger on the image.
        for id_hand in range(Hand.num_detected_hands): # all hands
            id_lmk = Hand.INDEX_FINGER_TIP # 8
            landmark_point = Hand.get_landmark(id_hand, id_lmk) # get landmark
            cv2.circle(image, landmark_point[:2], 1, (0, 0, 255), 2) # draw landmark
            # write text on the image
            txt = '({:d}, {:d})'.format(landmark_point[0], landmark_point[1])
            tip_point_for_text = (landmark_point[0]-20, landmark_point[1]-20)
            cv2.putText(image, org=tip_point_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_4)
    ```
    ```python
    # [1] Draw the all landmarks on the image.
    draw_hand_landmarks_only_tip(flipped_frame, Hand)
    cv2.imshow(wname, flipped_frame)
    ```
    - `cv2.putText`
        ```python
        cv2.putText(
            image,
            org=tip_point_for_text, # coordinate
            text=txt, # text
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1, # text size
            color=(0, 0, 255), # red = (0,0,255), green=(0,255,0), blue=(255,0,0)
            thickness=2, # thickness of text
            lineType=cv2.LINE_4
            )
        ```
- :o: [Sample 2] How to show only TIP of the all fingers<br>
    <image src="../image/myhand_tip.jpg" width=25%>
    ```python
    def draw_hand_landmarks_only_tip(image, Hand):
        # Draw only TIP landmarks on the image.
        id_list_tip = [4, 8, 12, 16, 20]
        for id_hand in range(Hand.num_detected_hands): # all hands
            for id_lmk in id_list_tip: # only TIP landmarks
                landmark_point = Hand.get_landmark(id_hand, id_lmk) # get landmark
                cv2.circle(image, landmark_point[:2], 1, (0, 0, 255), 2) # draw landmark
    ```
#### :white_square_button: Samples
**:exclamation: Note that these programs must be placed in the same directory as `our MediaPipe Class file` to work.**
- HandLandmark
    - [myhand_specific_lmk.py](../sample/myhand_specific_lmk.py)<br>
    <image src="../image/myhand_specific_lmk.jpg" width=20%>
- PoseLandmark
    - [mypose_specific_lmk.py](../sample/mypose_specific_lmk.py)<br>
    <image src="../image/mypose_specific_lmk.jpg" width=20%>
- FaceLandmark
    - [myface_specifics_lmk.py](../sample/myface_specifics_lmk.py)<br>
    <image src="../image/myface_specific_lmk.jpg" width=20%>

### :red_square: How to calcurate center of gravity (cog) of specific landmarks
#### :white_square_button: on `HandLandmarker`
<image src="../image/myhand_cog_of_tip.jpg" width=25%><br>
```python
# center of gravity
def draw_cog_point_of_all_tips(image, Hand):
    for id_hand in range(Hand.num_detected_hands): # all hands
        pt_cog = np.zeros((3,), dtype=int) # make initialized array: np.array([0, 0, 0])
        id_list_tip = [4, 8, 12, 16, 20]
        for id_lmk in id_list_tip:
            pt_cog += Hand.get_landmark(id_hand, id_lmk)
        pt_cog = (pt_cog/len(id_list_tip)).astype(int)
        cv2.circle(image, pt_cog[:2], 5, (0, 0, 255), 2) # draw landmark
```
#### :white_square_button: Samples
**:exclamation: Note that these programs must be placed in the same directory as `our MediaPipe Class file` to work.**
- HandLandmark
    - [myhand_center_of_gravity.py](../sample/myhand_center_of_gravity.py)
- PoseLandmark
    - [mypose_center_of_gravity.py](../sample/mypose_center_of_gravity.py)<br>
    <image src="../image/mypose_center_of_gravity.jpg" width=20%>
- FaceLandmark
    - [myface_center_of_gravity.py](../sample/myface_center_of_gravity.py)<br>
    <image src="../image/myface_center_of_gravity.jpg" width=20%>

### :red_square: How to make landmark array
- If you want to use the all landmark data as an list for all your processing, the following sample will help.
#### :white_square_button: on `HandLandmark`
- :o: function sample
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
- :o: How to use
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

### :red_square: How to calculate the angle between 2 vectors
- :o: function `calc_angle`: calculate the angle between 2 vectors
    ```python
    def calc_angle(v1, v2):
        v1_n = np.linalg.norm(v1)
        v2_n = np.linalg.norm(v2)
        cos_theta = np.inner(v1, v2) / (v1_n * v2_n)
        return np.rad2deg(np.arccos(cos_theta))
    ```
#### :white_square_button: on `HandLandmark`
- :o:[Sample 1] How to check the open/bend of the index finger<br>
    <image src="../image/myhand_open.jpg" width=25%><image src="../image/myhand_bend.jpg" width=25%><br>
    ```python
    def draw_open_bend_indexfinger(image, Hand):
        for id_hand in range(Hand.num_detected_hands):
            # pickup landmark points of index finger
            pt_ifmcp = Hand.get_landmark(id_hand, Hand.INDEX_FINGER_MCP)
            pt_ifpip = Hand.get_landmark(id_hand, Hand.INDEX_FINGER_PIP)
            pt_ifdip = Hand.get_landmark(id_hand, Hand.INDEX_FINGER_DIP)

            # draw index finger (MCP - PIP - DIP)
            cv2.circle(image, pt_ifmcp[:2], 5, (0, 0, 255), 3)
            cv2.circle(image, pt_ifpip[:2], 5, (0, 0, 255), 3)
            cv2.circle(image, pt_ifdip[:2], 5, (0, 0, 255), 3)
            cv2.line(image, pt_ifmcp[:2], pt_ifpip[:2], (0, 255, 0))
            cv2.line(image, pt_ifpip[:2], pt_ifdip[:2], (0, 255, 0))

            vec1 = pt_ifmcp - pt_ifpip # vector (pip -> mcp)
            vec2 = pt_ifdip - pt_ifpip # vector (pip -> dip)
            if calc_angle(vec1, vec2) > 140:
                txt = 'open'
            else:
                txt = 'bend'
            pt_for_text = (pt_ifmcp[0]+10, pt_ifmcp[1])
            cv2.putText(image, org=pt_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_4)
    ```
- :o:[Sample 2]How to check the 2d-angle between the vertical upward direction and the direction pointed by the index finger
    <image src="../image/myhand_angle_against_y.jpg" width=25%><br>
    ```python
    def draw_finger_angle_against_Y_axis(image, Hand):
        for id_hand in range(Hand.num_detected_hands):
            pt_iftip = Hand.get_landmark(id_hand, Hand.INDEX_FINGER_TIP)
            pt_ifpip = Hand.get_landmark(id_hand, Hand.INDEX_FINGER_PIP)

            # draw index finger (PIP - TIP)
            cv2.circle(image, pt_ifpip[:2], 5, (0, 0, 255), 3)
            cv2.circle(image, pt_iftip[:2], 5, (0, 0, 255), 3)
            cv2.line(image, pt_ifpip[:2], pt_iftip[:2], (0, 255, 0))

            vec1 = pt_pip - pt_mcp # 3d vector (pip -> mcp)
            vec2 = (0, -1) # 2d vector (vertical upward direction)
            angle = calc_angle(vec1[:2], vec2) # vec1 has 3-dimension
            if pt_tip[0] - pt_pip[0] < 0:
                angle = 360 - angle

            txt = '{:d}'.format(int(angle))
            pt_for_text = (pt_ifmcp[0]+10, pt_ifmcp[1])
            cv2.putText(image, org=pt_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_4)
    ```
#### :white_square_button: Samples
**:exclamation: Note that these programs must be placed in the same directory as `our MediaPipe Class file` to work.**
- HandLandmark
    - [mypose_calc_angle.py](../sample/myhand_calc_angle.py)
- PoseLandmark
    - [mypose_calc_angle.py](../sample/mypose_calc_angle.py)<br>
    <image src="../image/mypose_calc_angle_2lines.jpg" width=20%><image src="../image/mypose_calc_angle_Yaxis.jpg" width=20%>
- FaceLandmark
    - [myface_calc_angle_Yaxis.py](../sample/myface_calc_angle_Yaxis.py)<br>
    <image src="../image/myface_calc_angle_Yaxis.jpg" width=20%>

---
---
## :green_square: Original contents of each class
### :red_square: `HandLandmark`
- `HandLandmark` can discriminate between left and right hand.
- The following is sample function to draw with different colors for the left and right hands.
#### :o:[Sample] Draw with different colors for the left and right hands
<image src="../image/myhand_left.jpg" width=25%><image src="../image/myhand_right.jpg" width=25%><br>
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
#### :white_square_button: Samples
**:exclamation: Note that these programs must be placed in the same directory as `our MediaPipe Class file` to work.**
- [myhand_handedness.py](../sample/myhand_handedness.py)

---
### :red_square: `FaceLandmark`
- The following is sample to determine `left` or `right` according to the orientation of the face.
- There are several judgment methods, but the simple one is a judgment method that compares the x-coordinates of several landmarks.<br>
    <image src="../image/myface_left_right.jpg" width="25%" height="25%"><br>
#### :o:[Sample]Determine `left` or `right` orientation of the face
- sample function
    ```python
    def draw_left_right_with_face(image, Face):
        for id_face in range(Face.num_detected_faces):
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
                txt = 'right'
            elif 5*abs(l) < abs(r):
                txt = 'left'
            else:
                txt = ''
            pt_for_text = (pt_top[0]+10, pt_top[1])
            cv2.putText(image, org=pt_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_4)
    ```
#### :white_square_button: Samples
**:exclamation: Note that these programs must be placed in the same directory as `our MediaPipe Class file` to work.**
- [myface_left_right.py](../sample/myhand_left_right.py)

---
### :red_square: `FaceDetection`
- `FaceDetection` detects all faces in the input image and returns the 2d coordinates of the main key points.<br>
    <image src="../image/myface_detection.jpg" width="25%" height="25%"><br>
    - keypoints
        - `LEFT_EYE = 0`
        - `RIGHT_EYE = 1`
        - `NOSE_TIP = 2`
        - `MOUTH = 3`
        - `LEFT_EYE_TRAGION = 4`
        - `RIGHT_EYE_TRAGION = 5`
#### :o:[Sample]Show all keypoint and bounding box
- sample function
    ```python
    def draw_face_keypoints_boundingbox(image, FaceDtc):
        for id_face in range(FaceDtc.num_detected_faces):
            bx, by, bw, bh = FaceDtc.get_bounding_box(id_face)
            cv2.rectangle(image, (bx, by), (bx+bw, by+bh), (0,255,0), 2)
            for id_keypoint in range(FaceDtc.num_landmarks):
                keypoint = FaceDtc.get_landmark(id_face, id_keypoint)
                cv2.circle(image, tuple(keypoint), 2, (0, 0, 255), 3)
    ```
#### :white_square_button: Samples
**:exclamation: Note that these programs must be placed in the same directory as `our MediaPipe Class file` to work.**
- [myface_detection.py](../sample/myface_detection.py)

---
### :red_square: `PoseLandmark`
- `PoseLandmark` frequently produces landmarks of poor quality due to the difficulty of fitting the body into the camera's angle of view. You should be aware of that in your application development.
- `PoseLandmark` also outputs a segmentation mask for each detected person.
#### :o:[Sample 1] Calculate center of gravity of only visible landmarks
- MediaPipe also maintains low-confidence data for off-screen landmarks.
    - In the following image, the centers of gravity of the nose, left shoulder, right shoulder, left wrist, and right wrist are circled in blue, and those of only the coordinates inside the screen are circled in red.
    - The blue circle is positioned below the red circle because the wrists are expected to be at the bottom of the screen.<br>
    <image src="../image/mypose_cog_only_visible.jpg" width="25%" height="25%"><br>
- sample function
    ```python
    def draw_cog_point_of_only_visible_basic_points(image, Pose):
        for id_pose in range(Pose.num_detected_poses): # all poses
            pt_cog = np.zeros((3,), dtype=int) # make initialized array: np.array([0, 0, 0])
            id_list = [0, 11, 12, 15, 16] # nose, left shoulder, right shoulder, left wrist, right wrist
            cnt = 0
            for id_lmk in id_list:
                if Pose.get_landmark_visibility(id_pose, id_lmk)>0.5:
                    pt_cog += Pose.get_landmark(id_pose, id_lmk)
                    cnt += 1
            pt_cog = (pt_cog/cnt).astype(int)
            cv2.circle(image, pt_cog[:2], 5, (0, 0, 255), 2) # draw landmark
    ```
- you can also use presence score.
    ```python
    if Pose.get_landmark_visibility(id_pose, id_lmk) > 0.5 and Pose.get_landmark_presence(id_pose, id_lmk) > 0.5:
    ```
#### :o:[Sample 2] Determine which hand is up
- Display `right` when you raise your right hand and `left` when you raise your left hand. In addition, display `both` when you raise your both hands.<br>
    <image src="../image/mypose_judge_hand_up.jpg" width="25%" height="25%"><br>
- sample function
    ```python
    def draw_judge_hand_up(image, Pose):
        for id_pose in range(Pose.num_detected_poses): # all poses
            # keypoints
            pt_nose = Pose.get_landmark(id_pose, Pose.NOSE) # 0
            pt_left_index = Pose.get_landmark(id_pose, Pose.LEFT_INDEX) # 19
            pt_right_index = Pose.get_landmark(id_pose, Pose.RIGHT_INDEX) # 20

            if pt_nose[1] > pt_right_index[1] and pt_nose[1] > pt_left_index[1]:
                txt = 'both'
            elif pt_nose[1] > pt_left_index[1]:
                txt = 'left'
            elif pt_nose[1] > pt_right_index[1]:
                txt = 'right'
            else:
                txt = ''
            cv2.putText(image, org=pt_nose[:2], text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2, lineType=cv2.LINE_4)
    ```
#### :o:[Sample 3] Determine `O` or `X` of arm shape
- Make a shape of "O" or "X" with your arm and display it on the screen according to the shape.
- It is possible to judge by comparing the x-coordinate and y-coordinate values between multiple landmarks.<br>
    <image src="../image/mypose_judge_O_X.jpg" width="25%" height="25%">
- sample code
    ```python
    def draw_judge_O_X_with_pose(image, Pose):
        for id_pose in range(Pose.num_detected_poses): # all poses
            # keypoints of arms
            pt_left_shoulder = Pose.get_landmark(id_pose, Pose.LEFT_SHOULDER) # 11
            pt_right_shoulder = Pose.get_landmark(id_pose, Pose.RIGHT_SHOULDER) # 12
            pt_left_elbow = Pose.get_landmark(id_pose, Pose.LEFT_ELBOW) # 13
            pt_right_elbow = Pose.get_landmark(id_pose, Pose.RIGHT_ELBOW) # 14
            pt_left_wrist = Pose.get_landmark(id_pose, Pose.LEFT_WRIST) # 15
            pt_right_wrist = Pose.get_landmark(id_pose, Pose.RIGHT_WRIST) # 16

            txt = ''
            if (pt_right_elbow[1] > pt_left_wrist[1]
                and pt_right_elbow[0] < pt_left_elbow[0]
                and pt_right_elbow[0] < pt_right_wrist[0]
                and pt_left_elbow[1] > pt_right_wrist[1]
                and pt_left_wrist[0] < pt_right_wrist[0]
                ):
                txt = 'X'
            elif pt_right_shoulder[1] > pt_right_elbow[1] > pt_right_wrist[1]:
                if pt_left_shoulder[1] > pt_left_elbow[1] > pt_left_wrist[1]:
                    if pt_right_shoulder[0] < pt_right_wrist[0] < pt_left_wrist[0] < pt_left_shoulder[0]:
                        txt = 'O'
            pt_nose = Pose.get_landmark(id_pose, Pose.NOSE) # 0
            cv2.putText(image, org=pt_nose[:2], text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2, lineType=cv2.LINE_4)
    ```
#### :o:[Sample 4] Show segmentation mask
- The following sample code reduces the brightness outside of the detected person area.
- The pixel value of the segmentation mask represents the confidence score `[0, 1]` of personhood.<br>
    <image src="../image/mypose_segmentation.jpg" width="25%" height="25%"><br>
    - Transparency represents the conficence score.
- sample code
    ```python
    def visualize_all_segmentation_mask(image, Pose):
        all_seg_mask = Pose.get_all_segmentation_masks()

        normalized_seg_mask = all_seg_mask.astype(float)/np.max(all_seg_mask) # normalize [0.0, 1.0]
        mask = np.tile(normalized_seg_mask[:,:,None], [1,1,3])*0.7 + 0.3
        return (image * mask).astype(np.uint8)
    ```
    ```python
        annotated_frame = frame.copy()
        if Pose.num_detected_poses > 0:
            annotated_frame = visualize_all_segmentation_mask(frame, Pose)

        cv2.imshow(wname, annotated_frame) #### annotated frame
    ```
- You can binarize the segmentation mask by using a threshold value (e.g. `0.5`).
#### :white_square_button: Samples
**:exclamation: Note that these programs must be placed in the same directory as `our MediaPipe Class file` to work.**
- [mypose_cog_only_visible.py](../sample/mypose_cog_only_visible.py)
- [mypose_judge_hand_up.py](../sample/mypose_judge_hand_up.py)
- [mypose_judge_O_X.py](../sample/mypose_judge_O_X.py)
- [mypose_segmentation.py](../sample/mypose_segmentation.py)

---
### :red_square: `ImageSegmentation`
- `ImageSegmentation` returns the segmentation mask. Its pixel value represents the corresponding segmentation ID.
- In the `selfie_multiclass_256x256.tflite` model, it is as follows.
    - `BACKGROUND = 0`
    - `HAIR = 1`
    - `BODY_SKIN = 2`
    - `FACE_SKIN = 3`
    - `CLOTHES = 4`
    - `OTHERS = 5`<br>
<image src="../image/myseg.jpg" width=50% height=50%>
#### :o:[Sample] Show segmentation mask for selfie
- sample code
    ```python
    segmented_masks = Seg.get_segmentation_masks()
    # face skin pixels have 'True', others have 'False')
    face_skin_mask_binary = (segmented_masks == Seg.FACE_SKIN)

    # getter (face skin pixels have '255', others have '0')
    face_skin_mask = Seg.get_segmentation_mask(Seg.FACE_SKIN)
    ```
- You can get the confidence for the segmentation mask of each segmentation ID as a mask.
    ```python
    # Range [0.0, 1.0]
    face_skin_confidence_mask = Seg.get_confidence_mask(Seg.FACE_SKIN)
    ```
#### :white_square_button: Samples
**:exclamation: Note that these programs must be placed in the same directory as `our MediaPipe Class file` to work.**
- [myseg.py](../sample/myseg.py)

---
### :red_square: `ObjectDetection`
- `ObjectDetection` returns the category name and bounding box of the detected object.
    - [target object list](https://storage.googleapis.com/mediapipe-tasks/object_detector/labelmap.txt)<br>
<image src="../image/myobj.jpg" width=25% height=25%>
#### :o:[Sample] Show object's name and bounding box
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
- We recommend setting the threshold to 0.3 or more.
    ```python
    # make instance of our mediapipe class
    # you can set options
    Obj = ObjDtc(score_threshold=0.3)
    ```
#### :white_square_button: Samples
**:exclamation: Note that these programs must be placed in the same directory as `our MediaPipe Class file` to work.**
- [myobj.py](../sample/myobj.py)

# myhand_ges_simple_image.py
import cv2
from MediapipeHandGestureRecognition import MediapipeHandGestureRecognition as HandGesRec

img = cv2.imread("./img/standard/Balloon.bmp")
HandGes = HandGesRec(mode="image")
HandGes.detect(img)
if HandGes.num_detected_hands>0:
    print(HandGes.get_gesture(0), HandGes.get_score_gesture(0))
annotated_img = HandGes.visualize(img)

cv2.imshow("annotated_img", annotated_img)
cv2.waitKey(0)
HandGes.release()
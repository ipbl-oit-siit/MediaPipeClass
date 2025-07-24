# myhand_simple_image.py
import cv2
from MediapipeHandLandmark import MediapipeHandLandmark as HandLmk

img = cv2.imread("./img/standard/Balloon.bmp")
Hand = HandLmk(mode="image")
results = Hand.detect(img)
annotated_img = Hand.visualize(img)
cv2.imshow("annotated_img", annotated_img)
cv2.waitKey(0)
Hand.release()


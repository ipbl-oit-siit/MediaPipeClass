# mypose_simple_image.py
import cv2
from MediapipePoseLandmark import MediapipePoseLandmark as PoseLmk

img = cv2.imread("./img/standard/Balloon.bmp")
Pose = PoseLmk(mode="image")
Pose.detect(img)
annotated_img = Pose.visualize(img)
cv2.imshow("annotated_img", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
Pose.release()
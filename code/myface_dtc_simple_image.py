# myface_dtc_simple_image.py
import cv2
from MediapipeFaceDetection import MediapipeFaceDetection as FaceDect

img = cv2.imread("./img/standard/Balloon.bmp")
Face = FaceDect(mode="image")
Face.detect(img)
annotated_img = Face.visualize(img)
cv2.imshow("annotated_img", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
Face.release()
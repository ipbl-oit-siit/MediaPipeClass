# myobj_simple_image.py
import cv2
from MediapipeObjectDetection import MediapipeObjectDetection as ObjDetection

img = cv2.imread("./img/standard/Balloon.bmp")
Obj = ObjDetection(mode="image", score_threshold=0.5)
Obj.detect(img)
print(Obj.num_detected_objects)
annotated_frame = Obj.visualize(img)
cv2.imshow('annotated frame', annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
Obj.release()
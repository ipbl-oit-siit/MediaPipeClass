# myseg_simple_image.py
import cv2
from MediapipeImageSegmentation import MediapipeImageSegmentation as ImgSeg

img = cv2.imread("./img/standard/Balloon.bmp")
Seg = ImgSeg(mode="image")
Seg.detect(img)
normalized_masks = Seg.get_normalized_masks()
cv2.imshow('multiclass mask', cv2.applyColorMap(normalized_masks, cv2.COLORMAP_JET))
face_skin_masks = Seg.get_segmentation_mask(Seg.FACE_SKIN)
cv2.imshow('face skin', face_skin_masks)
cv2.imshow('frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
Seg.release()
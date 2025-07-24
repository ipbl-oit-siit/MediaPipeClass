import os
import urllib.request
import time
import numpy as np
# https://github.com/opencv/opencv/issues/17687
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mediapipe as mp

# https://developers.google.com/mediapipe/solutions/vision/image_segmenter
class MediapipeImageSegmentation():
    model_folder_path = './learned_models/mediapipe'
    # https://storage.googleapis.com/mediapipe-models/

    def __init__(
            self,
            model_folder_path=model_folder_path,
            base_url="",
            model_name="",
            model="selfie",
            output_category_mask=True,
            output_confidence_masks=True,
            mode="image"
            ):
        
        if base_url == "" or model_name == "":
            print("set Default "+model+" model")
            self.serect_model(model)

        self.mode = mode
        if self.mode == "image":
            rmode = mp.tasks.vision.RunningMode.IMAGE
        else:
            rmode = mp.tasks.vision.RunningMode.VIDEO

        model_path = self.set_model(self.base_url, model_folder_path, self.model_name)
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=rmode,
            output_category_mask=output_category_mask,
            output_confidence_masks=output_confidence_masks
        )
        self.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)

    def serect_model(self, model):
        if model == "selfie":
            # skin_type (selfie_multiclass_256x256.tflite)
            self.base_url = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/'
            self.model_name = 'selfie_multiclass_256x256.tflite'
            self.NUM_TYPES = 6
            self.BACKGROUND = 0
            self.HAIR = 1
            self.BODY_SKIN = 2
            self.FACE_SKIN = 3
            self.CLOTHES = 4
            self.OTHERS = 5
        elif model == "deeplab":
            self.base_url = 'https://storage.googleapis.com/mediapipe-tasks/image_segmenter/'
            self.model_name = 'deeplabv3.tflite'
            self.NUM_TYPES = 21
            self.BACKGROUND = 0
            self.AIRPLANE = 1
            self.BICYCLE = 2
            self.BIRD = 3
            self.BOAT = 4
            self.BOTTLE = 5
            self.BUS = 6
            self.CAR = 7
            self.CAT = 8
            self.CHAIR = 9
            self.COW = 10
            self.DINING_TABLE = 11
            self.DOG = 12
            self.HORSE = 13
            self.MOTORBIKE = 14
            self.PERSON = 15
            self.POTTED_PLANT = 16
            self.SHEEP = 17
            self.SOFA = 18
            self.TRAIN = 19
            self.TV_MONITOR = 20

    def set_model(self, base_url, model_folder_path, model_name):
        model_path = model_folder_path+'/'+model_name
        # download model if model file does not exist
        if not os.path.exists(model_path):
            # make directory if model_folder directory does not exist
            os.makedirs(model_folder_path, exist_ok=True)
            # download model file
            url = base_url+model_name
            save_name = model_path
            urllib.request.urlretrieve(url, save_name)
        return model_path

    def detect(self, img):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if self.mode == "image":
            self.results = self.segmenter.segment(mp_image)
        else:
            self.results = self.segmenter.segment_for_video(mp_image, int(time.time() * 1000))
        return self.results

    def get_segmentation_masks(self):
        return self.results.category_mask.numpy_view()

    def get_segmentation_mask(self, skin_type):
        masks = self.get_segmentation_masks()
        return 255*(masks == skin_type).astype(np.uint8)

    def get_confidence_mask(self, skin_type):
        return self.results.confidence_masks[skin_type].numpy_view()

    def get_normalized_masks(self):
        mask = self.results.category_mask.numpy_view()
        return (255.0*mask/(self.NUM_TYPES-1)).astype(np.uint8)

    def release(self):
        self.segmenter.close()

def main():
    cap = cv2.VideoCapture(0)
    model = "selfie"
    ImgSeg = MediapipeImageSegmentation(model=model)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            print("Ignoring empty camera frame.")
            break

        results = ImgSeg.detect(frame)

        normalized_masks = ImgSeg.get_normalized_masks()
        cv2.imshow('multiclass mask', cv2.applyColorMap(normalized_masks, cv2.COLORMAP_JET))

        # masks = ImgSeg.get_segmentation_masks()
        if model == "selfie":
            face_skin_masks = ImgSeg.get_segmentation_mask(ImgSeg.FACE_SKIN)
        elif model == "deeplab":
            face_skin_masks = ImgSeg.get_segmentation_mask(ImgSeg.PERSON)
        else:
            face_skin_masks = ImgSeg.get_segmentation_mask(0)
        cv2.imshow('face skin', face_skin_masks)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)&0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    ImgSeg.release()
    cap.release()

if __name__=='__main__':
    main()

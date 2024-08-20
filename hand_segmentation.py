# hand_segmentation.py

import os
import sys
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import segmentation_models as sm

class HandSegmenter:
    def __init__(self, model_path):
        self.model = load_model(model_path, custom_objects={'dice_loss_plus_1binary_focal_loss': sm.losses.DiceLoss()})
        self.image_shape = (720, 1280)  # The shape used during training

    def preprocess_image(self, image):
        image = image.resize(self.image_shape[::-1])
        image = np.array(image)
        image = image / 255.0
        return np.expand_dims(image, axis=0)

    def segment_hands(self, image):
        preprocessed = self.preprocess_image(image)
        prediction = self.model.predict(preprocessed)
        mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)
        return mask

    def get_hand_mask(self, image):
        original_size = image.size
        mask = self.segment_hands(image)
        mask = cv2.resize(mask, original_size[::-1], interpolation=cv2.INTER_NEAREST)
        return mask
# hand_segmentation.py

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

def unet(input_size=(256, 256, 3)):
    inputs = tf.keras.layers.Input(input_size)
    
    # Encoder (Downsampling)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bridge
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    
    # Decoder (Upsampling)
    up5 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv4)
    up5 = tf.keras.layers.Conv2D(256, 2, activation='relu', padding='same')(up5)
    merge5 = tf.keras.layers.concatenate([conv3, up5])
    conv5 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(merge5)
    conv5 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)
    
    up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)
    up6 = tf.keras.layers.Conv2D(128, 2, activation='relu', padding='same')(up6)
    merge6 = tf.keras.layers.concatenate([conv2, up6])
    conv6 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)
    
    up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
    up7 = tf.keras.layers.Conv2D(64, 2, activation='relu', padding='same')(up7)
    merge7 = tf.keras.layers.concatenate([conv1, up7])
    conv7 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)
    
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv7)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

class HandSegmenter:
    def __init__(self):
        self.model = unet(input_size=(256, 256, 3))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.image_shape = (256, 256)

    def preprocess_image(self, image):
        image = image.resize(self.image_shape)
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
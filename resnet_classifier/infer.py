import os

from PIL import Image
from PIL import ImageOps
import numpy as np

import tensorflow as tf

import resnet_classifier.roedeer_gender_classifier

target_size = 224

class RoedeerInference:
    def __init__(self, model_path):
        # Load the model
        self.model = tf.keras.models.load_model(model_path)

    def infer(self, pil_image, month):
        # Resize and pad input to target size
        pil_image = ImageOps.pad(pil_image, (target_size, target_size))

        x = tf.keras.utils.img_to_array(
            pil_image
        )

        x = x / 255.0

        x = tf.expand_dims(x, axis=0)

        month_tensor = np.array([month])

        one_hot = self.model((x, month_tensor), training=False)

        return np.argmax(one_hot), np.max(one_hot)






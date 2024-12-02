import numpy as np
import tensorflow as tf
from .base import DigitClassificationInterface

class CNNClassifier(DigitClassificationInterface):
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
    def predict(self, image: np.ndarray) -> int:
        if image.shape != (28, 28, 1):
            raise ValueError("CNN input must be 28x28x1 tensor")
        return int(np.argmax(self.model.predict(image[np.newaxis, ...])))
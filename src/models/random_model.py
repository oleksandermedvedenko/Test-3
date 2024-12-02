import numpy as np
from .base import DigitClassificationInterface

class RandomClassifier(DigitClassificationInterface):
    def predict(self, image: np.ndarray) -> int:
        if image.shape != (10, 10):
            raise ValueError("Random classifier input must be 10x10 array")
        return np.random.randint(0, 10)
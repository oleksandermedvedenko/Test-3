import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from .base import DigitClassificationInterface

class RandomForestClassifier(DigitClassificationInterface):
    def __init__(self):
        self.model = SklearnRF(n_estimators=100, random_state=42)
        
    def predict(self, image: np.ndarray) -> int:
        if image.size != 784:
            raise ValueError("RF input must be flattened 784-length array")
        flattened = image.reshape(1, -1)
        return int(self.model.predict(flattened)[0])
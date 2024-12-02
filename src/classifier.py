import numpy as np
from .models.cnn_model import CNNClassifier
from .models.rf_model import RandomForestClassifier
from .models.random_model import RandomClassifier

class DigitClassifier:
    def __init__(self, algorithm: str):
        self.algorithms = {
            "cnn": CNNClassifier,
            "rf": RandomForestClassifier,
            "rand": RandomClassifier
        }
        if algorithm not in self.algorithms:
            raise ValueError(f"Algorithm must be one of: {list(self.algorithms.keys())}")
        self.model = self.algorithms[algorithm]()
    
    def predict(self, image: np.ndarray) -> int:
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be numpy array")
        return self.model.predict(image)
from abc import ABC, abstractmethod
import numpy as np

class DigitClassificationInterface(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray) -> int:
        """
        Predict digit from input image
        Args:
            image: numpy array with model-specific shape
        Returns:
            int: predicted digit (0-9)
        """
        pass
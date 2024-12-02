import numpy as np
import pytest
from src.classifier import DigitClassifier

def test_cnn_classifier():
    classifier = DigitClassifier("cnn")
    image = np.random.rand(28, 28, 1)
    prediction = classifier.predict(image)
    assert isinstance(prediction, int)
    assert 0 <= prediction <= 9

def test_rf_classifier():
    classifier = DigitClassifier("rf")
    image = np.random.rand(28, 28)
    prediction = classifier.predict(image)
    assert isinstance(prediction, int)
    assert 0 <= prediction <= 9

def test_random_classifier():
    classifier = DigitClassifier("rand")
    image = np.random.rand(10, 10)
    prediction = classifier.predict(image)
    assert isinstance(prediction, int)
    assert 0 <= prediction <= 9

def test_invalid_algorithm():
    with pytest.raises(ValueError):
        DigitClassifier("invalid")
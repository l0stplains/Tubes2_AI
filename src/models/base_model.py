from abc import ABC, abstractmethod
import pickle

# TODO: tambahin docstrings, numpy style
# TODO: tambahin typing

class BaseModel(ABC):
    """
    Abstract base class for the machine learning models.
    """

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model


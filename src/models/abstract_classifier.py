from abc import ABC, abstractmethod

class AbstractClassifier(ABC):
    def __init__(self):
        self.model = None  # PyTorch model
        self.criterion = None  # Loss function
        self.optimizer = None  # Optimizer
        self.device = None

    @abstractmethod
    def train(self, train_loader):
        """
        Train the model.
        """
        pass

    @abstractmethod
    def validate(self, val_loader):
        """
        Validate the model.
        """
        pass

    @abstractmethod
    def predict(self, test_loader):
        """
        Make predictions with the model.
        """
        pass

    @abstractmethod
    def save_model(self, path):
        """
        Save the trained model.
        """
        pass

    @abstractmethod
    def load_model(self, path):
        """
        Load a trained model.
        """
        pass

    # Any additional methods needed for the classifier

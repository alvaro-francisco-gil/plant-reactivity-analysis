import torch
import torch.nn as nn
import torch.optim as optim

from src.models.classifier import FeatureClassifier
from typing import Dict
import numpy as np

class FullyConnectedClassifier(FeatureClassifier):
    def __init__(self, name: str = "fully_connected", features: list = None, parameters: Dict = None):
        super().__init__(name, features, parameters)
        
        # Neural Network Architecture
        self.model = nn.Sequential(
            nn.Linear(in_features=parameters.get("input_size", 0), out_features=parameters.get("hidden_size", 50)),
            nn.ReLU(),
            nn.Linear(in_features=parameters.get("hidden_size", 50), out_features=parameters.get("output_size", 10)),
            nn.Softmax(dim=1)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=parameters.get("learning_rate", 0.001))

    def train(self, parameters: Dict, **kwargs) -> None:
        # Implement the training procedure
        pass  # Replace with actual training code

    def load(self, parameters: Dict = None, **kwargs) -> None:
        # Implement the model loading
        pass  # Replace with code to load a model

    def save(self, parameters: Dict = None, **kwargs) -> None:
        # Implement the model saving
        pass  # Replace with code to save the model

    def classify(self, parameters: Dict, **kwargs) -> np.array:
        # Implement the classification
        pass  # Replace with code to perform classification

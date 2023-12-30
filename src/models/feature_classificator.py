import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureClassificator(nn.Module):
    def __init__(self, num_features, num_classes):
        """
        Initialize the neural network.

        :param num_features: The number of input features (size of the input layer).
        :param num_classes: The number of output classes (size of the output layer).
        """
        super(FeatureClassificator, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)  # Second fully connected layer
        self.fc3 = nn.Linear(64, num_classes)  # Output layer

    def forward(self, x):
        """
        Forward pass through the network.

        :param x: The input tensor.
        :return: The output tensor.
        """
        x = F.relu(self.fc1(x))  # Activation function after first layer
        x = F.relu(self.fc2(x))  # Activation function after second layer
        x = self.fc3(x)  # No activation function after output layer
        return x

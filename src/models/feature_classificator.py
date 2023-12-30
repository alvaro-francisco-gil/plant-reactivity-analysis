import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class FeatureClassificator(nn.Module):
    def __init__(self, num_features, num_classes):
        """
        Initialize the FeatureClassificator model.

        :param num_features: Number of input features.
        :param num_classes: Number of output classes.
        """
        super(FeatureClassificator, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)           # Second fully connected layer
        self.fc3 = nn.Linear(64, num_classes)   # Output layer

    def forward(self, x):
        """
        Forward pass of the model.

        :param x: Input tensor.
        :return: Output tensor.
        """
        x = torch.relu(self.fc1(x))  # Activation function after first layer
        x = torch.relu(self.fc2(x))  # Activation function after second layer
        x = self.fc3(x)              # No activation function after output layer
        return x

    def train_model(self, train_loader, val_loader, num_epochs, learning_rate):
        """
        Train the model.

        :param train_loader: DataLoader for the training data.
        :param val_loader: DataLoader for the validation data.
        :param num_epochs: Number of epochs to train.
        :param learning_rate: Learning rate for the optimizer.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            for data, targets in train_loader:
                optimizer.zero_grad()
                outputs = self(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            self.validate(val_loader)

    def validate(self, val_loader):
        """
        Validate the model.

        :param val_loader: DataLoader for the validation data.
        """
        self.eval()  # Set the model to evaluation mode
        total = 0
        correct = 0
        with torch.no_grad():
            for data, targets in val_loader:
                outputs = self(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')

    def predict(self, test_loader):
        """
        Predict using the model.

        :param test_loader: DataLoader for the test data.
        :return: Predictions
        """
        self.eval()  # Set the model to evaluation mode
        predictions = []
        with torch.no_grad():
            for data in test_loader:
                outputs = self(data)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.tolist())
        return predictions

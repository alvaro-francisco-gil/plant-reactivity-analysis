
import torch
import torch.nn as nn
import torch.optim as optim

from models.abstract_classifier import AbstractClassifier

class FullyConnectedClassifier(nn.Module):
    def __init__(self, input_size, output_size, parameters):
        super().__init__()

        dense_units = int(parameters['dense_units'])
        dense_layers = int(parameters['dense_layers'])
        dropout_rate = float(parameters.get('dropout_rate', 0.5))
        learning_rate = float(parameters.get('learning_rate', 0.001))

        layers = []

        # Add hidden layers
        for i in range(dense_layers):
            if i == 0:
                layers.append(nn.Linear(input_size, dense_units))
            else:
                layers.append(nn.Linear(dense_units, dense_units))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(dense_units, output_size))

        self.model = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Early stopping attributes
        self.early_stopping_patience = int(parameters.get('early_stopping_patience', 5))
        self.early_stopping_counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def train_model(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0

            # Training phase
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                total_correct += (predicted == target).sum().item()
                total_samples += target.size(0)

                loss.backward()
                self.optimizer.step()

            avg_loss = total_loss / len(train_loader)
            avg_accuracy = total_correct / total_samples

            # Validation phase
            val_loss = self.validate(val_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}, Training Accuracy: {avg_accuracy:.4f}, Validation Loss: {val_loss:.4f}")

            # Early stopping check
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.early_stopping_counter = 0  # Reset counter if validation loss improves
            else:
                self.early_stopping_counter += 1  # Increment counter if no improvement
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print("Early stopping triggered")
                    self.early_stop = True
                    break  # Break out of the training loop

            # Optional: Save the model at each epoch or when you get a better validation loss
            # self.save_model(f'model_epoch_{epoch}.pth')

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += self.criterion(output, target).item()
        return total_loss / len(val_loader)

    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                data = batch[0].to(self.device)  # Correctly extract data from the batch tuple
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                predictions.extend(predicted.tolist())
        return predictions

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
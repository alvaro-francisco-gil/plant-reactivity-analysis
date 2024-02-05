from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from abc import ABC, abstractmethod


class BaseClassifier(ABC):
    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def update_parameters(self, new_params):
        self.parameters.update(new_params)

    def calculate_accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def calculate_precision(self, y_true, y_pred):
        return precision_score(y_true, y_pred, average='binary')

    def calculate_recall(self, y_true, y_pred):
        return recall_score(y_true, y_pred, average='binary')

    def get_confusion_matrix(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)

from models.base_classifier import BaseClassifier


class MLClassifier(BaseClassifier):
    def __init__(self, model, parameters=None):
        self.model = model
        self.parameters = parameters or {}
        if self.parameters:
            self.model.set_params(**self.parameters)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

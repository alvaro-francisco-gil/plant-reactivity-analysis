from models.base_classifier import BaseClassifier


class NNClassifier(BaseClassifier):
    def __init__(self, parameters=None):
        self.parameters = parameters or {}
        self.model = self.build_model()

    def build_model(self):
        # Placeholder for model building logic
        raise NotImplementedError("Must implement build_model method")

    def train(self, X, y):
        fit_params = self.parameters.get('fit', {})
        self.model.fit(X, y, **fit_params)

    def predict(self, X):
        return self.model.predict(X)

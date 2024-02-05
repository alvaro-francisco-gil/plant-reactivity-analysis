from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class RandomForestClassifierWrapper:
    def __init__(self, parameters):
        self.parameters = parameters
        self.model = RandomForestClassifier(**self.parameters)

    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)

        if X_val is not None and y_val is not None:
            val_predictions = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_predictions)
            print(f"Validation Accuracy: {val_accuracy:.4f}")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X, y):
        predictions = self.model.predict(X)
        accuracy = accuracy_score(y, predictions)
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy

    def save_model(self, path):
        import joblib
        joblib.dump(self.model, path)

    def load_model(self, path):
        import joblib
        self.model = joblib.load(path)

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, \
                             GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


class Experiment:
    def __init__(self, train_df, test_df, label_column):

        assert set(train_df.columns) == set(test_df.columns), \
            "Training and testing data must have the same columns"
        assert set(train_df[label_column].unique()) == set(test_df[label_column].unique()), \
            "Training and testing labels must contain the same classes"

        self.train_features = train_df.drop(columns=[label_column])
        self.train_labels = train_df[label_column]
        self.test_features = test_df.drop(columns=[label_column])
        self.test_labels = test_df[label_column]
        self.class_labels = np.unique(self.train_labels)
        self.results = []

    def get_metrics(self, true_labels, predictions):
        f1 = f1_score(true_labels, predictions, average='weighted')
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted')
        recall = recall_score(true_labels, predictions, average='weighted')
        return f1, accuracy, precision, recall

    def print_confusion_matrix(self, true_labels, predictions):
        cm = confusion_matrix(true_labels, predictions)
        print(cm)

    def run_model_experiment(self, model_name, param_combination, print_cm=False):
        model_class = self.get_model_class(model_name)

        # Ensure 'random_state' is set to a fixed value for all models
        fixed_random_state = 42  # Fixed random state for reproducibility
        if 'random_state' in model_class().get_params().keys():
            # Update 'param_combination' to include the fixed 'random_state',
            # overriding it if it was previously specified.
            param_combination['random_state'] = fixed_random_state

        model = model_class(**param_combination)
        model.fit(self.train_features, self.train_labels)
        predictions = model.predict(self.test_features.to_numpy())

        f1, accuracy, precision, recall = self.get_metrics(self.test_labels, predictions)
        if print_cm:
            self.print_confusion_matrix(self.test_labels, predictions)

        self.results.append({
            "model_name": model_name,
            "parameters": param_combination,
            "f1": f1,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall
        })

    def run_all_models(self, classifier_par_dict, print_cm=False):
        for model_name, params in classifier_par_dict.items():
            for param_combination in ParameterGrid(params):
                print(f"Running experiments for {model_name} with params: {param_combination}")
                self.run_model_experiment(model_name, param_combination, print_cm)

    def get_model_class(self, model_name):
        model_classes = {
            "svm": SVC,
            "svm_rbf": SVC,
            "randomforest": RandomForestClassifier,
            "gradientboosting": GradientBoostingClassifier,
            "extratrees": ExtraTreesClassifier,
            "gaussiannb": GaussianNB,
            "adaboost": AdaBoostClassifier,
            "logisticregression": LogisticRegression,
            "kneighbors": KNeighborsClassifier,
            "lgbm": LGBMClassifier,
            "xgb": XGBClassifier
        }

        if model_name not in model_classes:
            raise ValueError(f"Unsupported model: {model_name}")
        return model_classes[model_name]

    def print_best_result_by_metric(self, metric):
        best_result = max(self.results, key=lambda x: x[metric])
        print(f"Best {metric}: {best_result[metric]} for model {best_result['model_name']} \
              with parameters {best_result['parameters']}")
        # Print other metrics as well
        print(f"Other Metrics: Accuracy: {best_result['accuracy']}, Precision: {best_result['precision']}, \
              Recall: {best_result['recall']}, F1: {best_result['f1']}")

    def print_best_result_by_model(self, metric):
        model_names = set(result['model_name'] for result in self.results)
        for model_name in model_names:
            model_results = [result for result in self.results if result['model_name'] == model_name]
            best_result = max(model_results, key=lambda x: x[metric])
            print(f"Best {metric} for {model_name}: {best_result[metric]} with parameters {best_result['parameters']}")

    def train_and_evaluate_model(self, model_name, param_combination):
        # Get the model class based on the model name
        model_class = self.get_model_class(model_name)
        # Initialize the model with the provided parameters
        model = model_class(**param_combination)
        # Train the model on the training data
        model.fit(self.train_features, self.train_labels)
        # Make predictions on the test data
        predictions = model.predict(self.test_features.to_numpy())
        # Calculate metrics
        f1, accuracy, precision, recall = self.get_metrics(self.test_labels, predictions)
        # Print confusion matrix
        self.print_confusion_matrix(self.test_labels, predictions)
        # Print metrics
        print(f"Metrics for {model_name} with params {param_combination}:")
        print(f"F1: {f1}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")

    def save_results_to_csv(self, filename="experiment_results.csv"):
        if not self.results:
            print("No results to save.")
            return

        # Convert results to a DataFrame
        results_df = pd.DataFrame(self.results,
                                  columns=['Model', 'Parameter', 'F1', 'Accuracy', 'Precision', 'Recall'])

        # Save the DataFrame to a CSV file
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

    @classmethod
    def from_arrays(cls, train_features, train_labels, test_features, test_labels):
        # Convert numpy arrays to pandas DataFrames
        train_df = pd.DataFrame(train_features)
        test_df = pd.DataFrame(test_features)
        # Assuming the labels are the last column after concatenating them to the features
        train_df['label'] = train_labels
        test_df['label'] = test_labels
        # Create an instance using the __init__ constructor
        return cls(train_df, test_df, 'label')

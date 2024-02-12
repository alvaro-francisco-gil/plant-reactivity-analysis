from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
import pandas as pd
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

    def get_metrics(self, Y, predictions):
        f1 = f1_score(Y, predictions, average='macro', zero_division=0)
        accuracy = accuracy_score(Y, predictions)
        precision = precision_score(Y, predictions, average='macro', zero_division=0)
        recall = recall_score(Y, predictions, average='macro', zero_division=0)
        # print(f'F1 Score: {f1}\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}')
        return f1, accuracy, precision, recall

    def print_confusion_matrix(self, Y, pred):
        # plt.figure()
        # cm = confusion_matrix(y_true=Y, y_pred=pred)
        # print(cm)
        pass

    def run_model_experiment(self, model_name, params):

        model_class = self.get_model_class(model_name)
        for param in params:
            # Determine the appropriate parameter for model instantiation
            if 'n_estimators' in model_class().get_params():
                model = model_class(n_estimators=param) if param is not None else model_class()
            elif 'C' in model_class().get_params():
                model = model_class(C=param) if param is not None else model_class()
            else:
                model = model_class()  # For models without these specific parameters

            model.fit(self.train_features, self.train_labels)
            predictions = model.predict(self.test_features)
            f1, accuracy, precision, recall = self.get_metrics(self.test_labels, predictions)
            self.print_confusion_matrix(self.test_labels, predictions)
            self.results.append([model_name, param, f1, accuracy, precision, recall])

    def run_all_models(self, classifier_par_dict):
        for model_name, params in classifier_par_dict.items():
            print(f"Running experiments for {model_name}")
            self.run_model_experiment(model_name, params)

    def get_model_class(self, model_name):
        model_classes = {
            "svm": SVC,
            "svm_rbf": SVC,  # Example, assuming you'd set kernel='rbf' in params if needed
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

    def print_best_result(self, metric='f1'):
        if not self.results:
            print("No results to display.")
            return

        # Define a dictionary to map metric names to their positions in the results
        metric_indices = {'f1': 2, 'accuracy': 3, 'precision': 4, 'recall': 5}

        # Check if the metric name is valid
        if metric not in metric_indices:
            print(f"Metric '{metric}' is not valid. Choose from {list(metric_indices.keys())}.")
            return

        # Find the result with the best value for the chosen metric
        best_result = max(self.results, key=lambda x: x[metric_indices[metric]])

        # Print or return the best result
        print(f"Best {metric} result:")
        print(f"Model: {best_result[0]}, Parameter: {best_result[1]}, "
              f"F1: {best_result[2]}, Accuracy: {best_result[3]}, "
              f"Precision: {best_result[4]}, Recall: {best_result[5]}")

    def print_best_result_by_model(self, metric='f1'):
        if not self.results:
            print("No results to display.")
            return

        # Define a dictionary to map metric names to their positions in the results
        metric_indices = {'f1': 2, 'accuracy': 3, 'precision': 4, 'recall': 5}

        # Check if the metric name is valid
        if metric not in metric_indices:
            print(f"Metric '{metric}' is not valid. Choose from {list(metric_indices.keys())}.")
            return

        # Group results by model
        model_groups = {}
        for result in self.results:
            model_name = result[0]
            if model_name not in model_groups:
                model_groups[model_name] = []
            model_groups[model_name].append(result)

        # For each model, find the best result based on the specified metric
        for model_name, results in model_groups.items():
            best_result = max(results, key=lambda x: x[metric_indices[metric]])
            print(f"Best {metric} result for {model_name}:")
            print(f"Parameter: {best_result[1]}, F1: {best_result[2]}, "
                  f"Accuracy: {best_result[3]}, Precision: {best_result[4]}, "
                  f"Recall: {best_result[5]}\n")

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

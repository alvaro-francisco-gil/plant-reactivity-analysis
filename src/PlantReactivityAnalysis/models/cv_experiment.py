from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


class ExperimentCV:
    def __init__(self, df, label_column, n_splits=5):
        self.features = df.drop(columns=[label_column])
        self.labels = df[label_column]
        self.class_labels = np.unique(self.labels)
        self.results = []
        self.n_splits = n_splits

    def get_metrics_cv(self, model, X, Y, n_splits):
        cv = StratifiedKFold(n_splits=n_splits)

        scores = {
            'f1': make_scorer(f1_score, average='macro', zero_division=0),
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='macro', zero_division=0),
            'recall': make_scorer(recall_score, average='macro', zero_division=0)
        }

        metrics_results = {}
        for metric_name, scorer in scores.items():
            cv_score = cross_val_score(model, X, Y, cv=cv, scoring=scorer)
            metrics_results[metric_name] = cv_score.mean()

        print(f"F1 Score: {metrics_results['f1']}, Accuracy: {metrics_results['accuracy']}, "
              f"Precision: {metrics_results['precision']}, Recall: {metrics_results['recall']}")
        return metrics_results

    def run_model_experiment(self, model_name, params):
        model_class = self.get_model_class(model_name)
        for param in params:
            if 'n_estimators' in model_class().get_params():
                model = model_class(n_estimators=param) if param is not None else model_class()
            elif 'C' in model_class().get_params():
                model = model_class(C=param) if param is not None else model_class()
            else:
                model = model_class()

            metrics_results = self.get_metrics_cv(model, self.features, self.labels, self.n_splits)
            self.results.append([model_name, param, metrics_results['f1'], metrics_results['accuracy'],
                                 metrics_results['precision'], metrics_results['recall']])

    def run_all_models(self, classifier_par_dict):
        for model_name, params in classifier_par_dict.items():
            print(f"Running experiments for {model_name}")
            self.run_model_experiment(model_name, params)

    def get_model_class(self, model_name):
        if model_name in ["svm", "svm_rbf"]:
            return SVC
        elif model_name == "randomforest":
            return RandomForestClassifier
        elif model_name == "gradientboosting":
            return GradientBoostingClassifier
        elif model_name == "extratrees":
            return ExtraTreesClassifier
        elif model_name == "gaussiannb":
            return GaussianNB
        else:
            raise ValueError(f"Unsupported model: {model_name}")

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

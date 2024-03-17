import numpy as np
import pandas as pd
import joblib
import os

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier)

from PlantReactivityAnalysis.data.get_dataset import get_dataset_by_question
from PlantReactivityAnalysis.config import FEATURES_LETTERS_DIR, MODELS_DIR, FEATURES_ONE_SEC_DIR, EXPERIMENT_DIR
from PlantReactivityAnalysis.models.parameters import PARAMETER_GRID, CORRELATION_TRESHOLDS


class Experiment:
    """
    A class to encapsulate the process of running machine learning experiments,
    including training, evaluating models, and saving results.
    """
    def __init__(self, train_df, test_df, label_column='target'):
        """
        Initialize the Experiment instance with training and testing data.
        """
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
        self.best_results = {}

    def get_model_class(self, model_name):
        """
        Retrieves a classifier class based on the provided model name.

        :param model_name: The name of the model to retrieve.
        :return: The classifier class associated with the `model_name`.
        :raises ValueError: If the `model_name` is not supported (i.e., not a key in `model_classes`).
        """
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

    def get_metrics(self, true_labels, predictions):
        """
        Computes and returns various evaluation metrics for classification predictions.

        :param true_labels: The true labels of the dataset.
        :param predictions: The predicted labels by the model.
        :return: A tuple containing the F1 score, accuracy, precision, and recall, all weighted by class.
        """
        f1 = f1_score(true_labels, predictions, average='weighted')
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted')
        recall = recall_score(true_labels, predictions, average='weighted')
        return f1, accuracy, precision, recall

    def train_and_evaluate_model(self, model_name, param_combination):
        """
        Trains a model specified by `model_name` with parameters `param_combination`, evaluates it on test data,
        and prints out the model's performance metrics.

        :param model_name: Name of the machine learning model to use.
        :param param_combination: Dictionary of parameters for the model.
        """
        # Initialize and train the model
        model = self.get_model_class(model_name)(**param_combination)
        model.fit(self.train_features, self.train_labels)

        # Evaluate the model
        predictions = model.predict(self.test_features.to_numpy())
        metrics = self.get_metrics(self.test_labels, predictions)

        # Display the performance metrics
        print(f"Metrics for {model_name} with params {param_combination}\
              : F1: {metrics[0]}, Accuracy: {metrics[1]}, Precision: {metrics[2]}, Recall: {metrics[3]}")

    def run_model_experiment(self, model_name, param_combination):
        """
        Executes a complete experiment for a given model and parameter combination, including
        training, evaluating, and storing results for comparison.

        :param model_name: The name of the model to be used in the experiment.
        :param param_combination: A dictionary of parameters for initializing the model.

        This method ensures reproducibility by setting a fixed random state if applicable,
        trains the model, makes predictions, computes metrics, and updates the best results
        and general results list with the experiment's outcomes.
        """
        model_class = self.get_model_class(model_name)

        fixed_random_state = 42
        if 'random_state' in model_class().get_params().keys():
            param_combination['random_state'] = fixed_random_state

        model = model_class(**param_combination)
        model.fit(self.train_features, self.train_labels)
        if model_name in ['kneighbors', 'xgb']:
            predictions = model.predict(self.test_features.to_numpy())
        else:
            predictions = model.predict(self.test_features)

        f1, accuracy, precision, recall = self.get_metrics(self.test_labels, predictions)
        cm = confusion_matrix(self.test_labels, predictions)

        # Update to store the model object if this is the best accuracy for the model
        if model_name not in self.best_results or self.best_results[model_name]['accuracy'] < accuracy:
            self.best_results[model_name] = {
                "model": model,
                "parameters": param_combination,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": cm
            }

        self.results.append({
            "model_name": model_name,
            "parameters": param_combination,
            "f1": f1,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": cm
        })

    def run_all_models(self, classifier_par_dict):
        """
        Iterates over all specified models and their parameter grids to run experiments.

        :param classifier_par_dict: Dictionary where keys are model names and values dictionaries of parameter grids.
        """
        # Iterate through each model and its parameter combinations
        for model_name, params in classifier_par_dict.items():
            for param_combination in ParameterGrid(params):
                print(f"Running experiments for {model_name} with params: {param_combination}")
                self.run_model_experiment(model_name, param_combination)

    def append_results_to_csv(self, file_path, parameters=None):
        """
        Appends experiment results to a CSV file, creating a new file if one doesn't exist.

        :param file_path: Path to the CSV file where results will be saved.
        :param parameters: Additional parameters to add to each row of results.
        """
        # Prepare results data with optional additional parameters
        expanded_results = []
        for result in self.results:
            expanded_row = result.copy()
            if parameters:
                expanded_row.update(parameters)
            expanded_results.append(expanded_row)

        # Convert to DataFrame for easier manipulation
        df_results = pd.DataFrame(expanded_results)

        try:
            # Attempt to append to existing file, or create new
            df_existing = pd.read_csv(file_path)
            df_updated = pd.concat([df_existing, df_results], ignore_index=True)
        except FileNotFoundError:
            print(f"No existing file found at {file_path}. Creating a new file.")
            df_updated = df_results

        # Save the consolidated results
        df_updated.to_csv(file_path, index=False)
        print(f"Results updated and saved to {file_path}.")

    def save_best_models(self, directory="best_models"):
        """
        Saves the best model from each category to disk.

        :param directory: Directory path where best models will be saved.
        """
        # Ensure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save each best model to the specified directory
        for model_name, result in self.best_results.items():
            filename = os.path.join(directory, f"best_{model_name}.joblib")
            joblib.dump(result['model'], filename)


def run_experiment_in_folder(features_folder, research_questions, correlation_tresholds,
                             models_parameters, results_file):
    for ct in correlation_tresholds:
        for file_path in os.listdir(features_folder):
            file = os.path.splitext(file_path)[0]
            parts_file = file.split('_')
            ws = float(parts_file[4][2:])
            hl = float(parts_file[5][2:])
            print('--------------------- Processing file: ', file, '----------------------------')

            datasets = get_dataset_by_question(path=FEATURES_LETTERS_DIR/file_path,
                                               rqs=research_questions, corr_threshold=ct)

            for rq in research_questions:
                print(f"  Processing RQ {rq}")

                train_df, test_df = datasets[rq]
                experiment = Experiment(train_df, test_df)
                experiment.run_all_models(models_parameters)

                # Store the results and models
                folder_rq = 'rq' + str(rq)
                model_folder = 'ws' + str(ws) + '_hl' + str(hl) + '_ct' + str(ct)
                experiment.save_best_models(directory=MODELS_DIR / folder_rq / model_folder)
                rows = {'RQ': rq, 'Window Size': ws, 'Hop Length': hl, 'Correlation Treshold': ct}
                experiment.append_results_to_csv(results_file, parameters=rows)


if __name__ == '__main__':

    models_parameters = PARAMETER_GRID
    results_file = EXPERIMENT_DIR / 'experiment_results.csv'

    # Research Questions 1 & 2
    research_questions = [1, 2]
    features_folder = FEATURES_LETTERS_DIR
    correlation_tresholds = CORRELATION_TRESHOLDS
    run_experiment_in_folder(features_folder, research_questions, correlation_tresholds,
                             models_parameters, results_file)

    # Research Question 5
    research_questions = [5]
    features_folder = FEATURES_ONE_SEC_DIR
    correlation_tresholds = [0.8]
    run_experiment_in_folder(features_folder, research_questions, correlation_tresholds,
                             models_parameters, results_file)

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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, \
                             GradientBoostingClassifier, AdaBoostClassifier


from PlantReactivityAnalysis.data.get_dataset import get_dataset_by_question
from PlantReactivityAnalysis.config import FEATURES_LETTERS_DIR, MODELS_DIR
import PlantReactivityAnalysis.models.parameters as param


class Experiment:
    def __init__(self, train_df, test_df, label_column='target'):

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

    def get_metrics(self, true_labels, predictions):
        f1 = f1_score(true_labels, predictions, average='weighted')
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted')
        recall = recall_score(true_labels, predictions, average='weighted')
        return f1, accuracy, precision, recall

    def run_model_experiment(self, model_name, param_combination):
        model_class = self.get_model_class(model_name)

        fixed_random_state = 42  # Fixed random state for reproducibility
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
                "model": model,  # Store the model object
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
        for model_name, params in classifier_par_dict.items():
            for param_combination in ParameterGrid(params):
                print(f"Running experiments for {model_name} with params: {param_combination}")
                self.run_model_experiment(model_name, param_combination)

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
        # Print metrics
        print(f"Metrics for {model_name} with params {param_combination}:")
        print(f"F1: {f1}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")

    def append_results_to_csv(self, file_path, parameters=None):

        # Process self.results to possibly include new parameters and expand existing ones
        expanded_results = []
        for result in self.results:
            expanded_row = result.copy()
            if parameters is not None:
                expanded_row.update(parameters)
            expanded_results.append(expanded_row)

        # Convert expanded results to DataFrame
        df_results = pd.DataFrame(expanded_results)

        try:
            # If file exists, load it and concatenate, else df_existing will just be df_results
            df_existing = pd.read_csv(file_path)
            df_updated = pd.concat([df_existing, df_results], ignore_index=True)
        except FileNotFoundError:
            print(f"No existing file found at {file_path}. A new file will be created.")
            df_updated = df_results

        # Save the updated DataFrame back to the CSV file
        df_updated.to_csv(file_path, index=False)
        print(f"Results updated and saved to {file_path}.")

    def save_best_models(self, directory="best_models"):
        if not os.path.exists(directory):
            os.makedirs(directory)

        for model_name, result in self.best_results.items():
            filename = os.path.join(directory, f"best_{model_name}.joblib")
            joblib.dump(result['model'], filename)
            # print(f"Saved {model_name} model to {filename}")

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


if __name__ == '__main__':

    rqs = [1, 2]

    for ct in param.CORRELATION_TRESHOLDS:
        for file_path in os.listdir(FEATURES_LETTERS_DIR):
            file = os.path.splitext(file_path)[0]
            parts_file = file.split('_')
            ws = float(parts_file[4][2:])
            hl = float(parts_file[5][2:])
            print('--------------------- Processing file: ', file, '----------------------------')

            datasets = get_dataset_by_question(path=FEATURES_LETTERS_DIR/file_path, rqs=rqs, corr_threshold=ct)
            results = {}

            for rq in rqs:
                print(f"  Processing RQ {rq}")

                train_df, test_df = datasets[rq]
                experiment = Experiment(train_df, test_df)
                experiment.run_all_models(param.PARAMETER_GRID_LETTERS)

                # Store the results and models
                folder_rq = 'rq' + str(rq)
                model_folder = 'ws' + str(ws) + '_hl' + str(hl) + '_ct' + str(ct)
                experiment.save_best_models(directory=MODELS_DIR / folder_rq / model_folder)
                rows = {'RQ': rq, 'Window Size': ws, 'Hop Length': hl, 'Correlation Treshold': ct}
                experiment.append_results_to_csv('experiment_results.csv', parameters=rows)

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


class Experiment:
    def __init__(self, train_df, test_df, label_column):
        assert set(train_df.columns) == set(test_df.columns), "Training and testing data must have the same columns"
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
        print(f'F1 Score: {f1}\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}')
        return f1, accuracy, precision, recall

    def print_confusion_matrix(self, Y, pred):
        plt.figure()
        cm = confusion_matrix(y_true=Y, y_pred=pred)
        print(cm)
        """
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, linewidths=0.5, linecolor='k', cbar=True)
        plt.figure(figsize=(10, 8))  # Adjust the figure size as necessary
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        # Dynamically set the tick labels
        unique_labels = np.unique(np.concatenate((Y, pred)))
        plt.xticks(np.arange(len(unique_labels)) + 0.5, unique_labels)
        plt.yticks(np.arange(len(unique_labels)) + 0.5, unique_labels, rotation=0)
        plt.tight_layout()
        plt.show()
        """

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

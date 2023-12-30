import pandas as pd
from torch.utils.data import Dataset
from scipy import stats

class FeaturesDataset(Dataset):

    def __init__(self, features: list, targets: list, feature_labels: list = None):
        """
        Initialization method for the FeaturesDataset.

        :param features: A list of lists, where each sublist represents the features of a single sample.
        :param targets: A list of targets corresponding to each sample in features.
        :param feature_labels: Optional list of labels for each feature. Length should match the length of a single feature vector.

        The features are stored in a pandas DataFrame for efficient data manipulation,
        while the targets are stored as a separate list.
        """
        assert len(features) == len(targets), "Features and targets must have the same length"
        self.features = pd.DataFrame(features, columns=feature_labels)
        self.targets = targets

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        :return: Integer representing the number of samples.
        """
        return len(self.features)

    def __getitem__(self, idx: int):
        """
        Retrieves a single sample and its corresponding target from the dataset.

        :param idx: The index of the sample to retrieve.
        :return: A tuple containing the feature vector and its corresponding target.
        """
        feature = self.features.iloc[idx].values.tolist()
        target = self.targets[idx]
        return feature, target

    def save_to_csv(self, filepath):
        """
        Saves the dataset to a CSV file.

        :param filepath: The path to the file where the dataset will be saved.
        """
        df = self.features.copy()
        df['target'] = self.targets
        df.to_csv(filepath, index=False)

    @classmethod
    def load_from_csv(cls, filepath, feature_labels=None):
        """
        Loads a dataset from a CSV file.

        :param filepath: The path to the CSV file to load the dataset from.
        :param feature_labels: Optional list of feature labels to apply to the DataFrame.
        :return: An instance of FeaturesDataset with data loaded from the CSV file.
        """
        df = pd.read_csv(filepath)
        targets = df['target'].tolist()
        features = df.drop('target', axis=1)
        return cls(features.values.tolist(), targets, feature_labels=feature_labels if feature_labels else features.columns.tolist())

    def get_targets(self):
        """
        Returns the targets of the dataset.

        :return: A list of targets.
        """
        return self.targets

    def get_features(self):
        """
        Returns the features of the dataset.

        :return: A list of lists, where each sublist represents the features of a single sample.
        """
        return self.features.values.tolist()

    def get_labels(self):
        """
        Returns the labels for the features in the dataset.

        :return: A list of feature labels.
        """
        return self.features.columns.tolist()

    def get_data(self):
        """
        Returns all the features and targets in the dataset.

        :return: A tuple containing two lists: the list of feature vectors and the list of targets.
        """
        return self.get_features(), self.get_targets()

    def remove_nan_columns(self):
        """
        Removes columns from the features DataFrame that contain NaNs and updates the feature labels accordingly.
        Prints the names of the removed columns.
        """
        cols_with_nan = self.features.columns[self.features.isna().any()].tolist()
        self.features.drop(columns=cols_with_nan, inplace=True)
        if cols_with_nan:
            print(f"Removed columns with NaNs: {cols_with_nan}")
        else:
            print("No columns with NaNs found.")

    def normalize_features(self, method='zscore'):
        """
        Normalizes the features in the dataset.

        :param method: The method of normalization. Supported methods include 'zscore' for z-score normalization
                       and 'minmax' for min-max normalization.
        """
        if method == 'zscore':
            # Z-score normalization
            self.features = (self.features - self.features.mean()) / self.features.std()
        elif method == 'minmax':
            # Min-max normalization
            self.features = (self.features - self.features.min()) / (self.features.max() - self.features.min())
        else:
            raise ValueError("Unsupported normalization method. Choose 'zscore' or 'minmax'.")
        
    def get_features_dataframe(self):
        """
        Returns the features as a pandas DataFrame.
        """
        return self.features

    def treat_outliers(self, iqr_multiplier=1.5):
        """
        Treats outliers in the dataset by replacing them with the limit value in the direction of the outlier.
        Outliers are determined based on the specified IQR multiplier.

        :param iqr_multiplier: The multiplier for the IQR to determine outliers. Defaults to 1.5.
        """
        Q1 = self.features.quantile(0.25)
        Q3 = self.features.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR

        # Replace outliers with the nearest boundary value
        for feature in self.features.columns:
            self.features[feature] = self.features[feature].mask(
                self.features[feature] < lower_bound[feature], lower_bound[feature])
            self.features[feature] = self.features[feature].mask(
                self.features[feature] > upper_bound[feature], upper_bound[feature])

        print(f"Outliers have been treated based on the {iqr_multiplier} * IQR criterion.")


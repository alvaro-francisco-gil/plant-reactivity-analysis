import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np

class FeaturesDataset(Dataset):

    def __init__(self, features: list, targets: list = None, feature_labels: list = None):
        """
        Initialization method for the FeaturesDataset.

        :param features: A list of lists, where each sublist represents the features of a single sample.
        :param targets: A list of targets corresponding to each sample in features.
        :param feature_labels: Optional list of labels for each feature. Length should match the length of a single feature vector.

        The features are stored in a pandas DataFrame for efficient data manipulation,
        while the targets are stored as a separate list.
        """
        #assert len(features) == len(targets), "Features and targets must have the same length"
        self.features = pd.DataFrame(features, columns=feature_labels)
        self.targets = targets
        self.processed = False

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
        feature = self.features.iloc[idx].values
        target = self.targets[idx]

        # Convert to PyTorch tensors
        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.long)  # Use torch.float32 if it's a regression task

        return feature_tensor, target_tensor

    # GETTERS AND SETTERS


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

    def get_feature_labels(self):
        """
        Returns the feature labels (column names) of the features DataFrame.
        """
        return self.features.columns.tolist()
        
    def get_features_dataframe(self):
        """
        Returns the features as a pandas DataFrame.
        """
        return self.features

    def head(self, n=5):
        """
        Displays the first n rows of the features DataFrame.

        :param n: The number of rows to display. Defaults to 5.
        """
        return self.features.head(n)
    
    def shape(self):
        """
        Returns the shape of the features DataFrame.
        """
        return self.features.shape
    
    # HANDLE DATA

    def drop_columns(self, columns_to_drop: list):
        """
        Drops specified columns from the features DataFrame.

        :param columns_to_drop: A list of column names to be dropped.
        """
        self.features = self.features.drop(columns=columns_to_drop, errors='ignore')


    def drop_rows(self, row_indices: list):
        """
        Drops rows from the features DataFrame and the corresponding elements from the targets list.

        :param row_indices: A list of indices of rows to be dropped.
        """
        # Drop rows from the features DataFrame
        self.features = self.features.drop(row_indices, errors='ignore').reset_index(drop=True)

        # Drop the corresponding targets
        self.targets = [target for idx, target in enumerate(self.targets) if idx not in row_indices]

    def split_dataset_in_loaders(self, test_size=0.3, val_size=0.5, random_state=42, batch_size = 32):
        """
        Splits the dataset into training, validation, and testing loaders.

        :param test_size: Proportion of the dataset to include in the test split.
        :param val_size: Proportion of the test set to include in the validation set.
        :param random_state: Random state for reproducibility.
        :return: A tuple (train_dataset, val_dataset, test_dataset).
        """
        dataset_size = len(self)
        indices = list(range(dataset_size))
        train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)
        val_indices, test_indices = train_test_split(test_indices, test_size=val_size, random_state=random_state)

        train_dataset = Subset(self, train_indices)
        val_dataset = Subset(self, val_indices)
        test_dataset = Subset(self, test_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, val_loader, test_loader

    # PROCESS DATA

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
            print('The Features were properly normalized using \'zscore\' method.')
        elif method == 'minmax':
            # Min-max normalization
            self.features = (self.features - self.features.min()) / (self.features.max() - self.features.min())
            print('The Features were properly normalized using \'minmax\' method.')
        else:
            raise ValueError("Unsupported normalization method. Choose 'zscore' or 'minmax'.")

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

        for feature in self.features.columns:
            # Count the number of outliers for the current feature
            outliers_count = ((self.features[feature] < lower_bound[feature]) | 
                            (self.features[feature] > upper_bound[feature])).sum()
            total_count = len(self.features[feature])
            outlier_percentage = (outliers_count / total_count) * 100

            # Replace outliers with the nearest boundary value
            self.features[feature] = self.features[feature].mask(
                self.features[feature] < lower_bound[feature], lower_bound[feature])
            self.features[feature] = self.features[feature].mask(
                self.features[feature] > upper_bound[feature], upper_bound[feature])

            print(f"Feature '{feature}': {outlier_percentage:.2f}% outliers treated.")

        print(f"Outliers have been treated based on the {iqr_multiplier} * IQR criterion.")

    def reduce_features(self, targets, corr_threshold=0.8):
        """
        Reduces features based on t-test/ANOVA and correlation matrix.

        :param targets: A list of class values corresponding to the targets.
        :param corr_threshold: Threshold for feature correlation. Features with correlation above this threshold
                               will be considered for removal based on their p-values.
        """
        # Perform t-test or ANOVA and store p-values
        p_values = []
        num_classes = len(set(targets))
        for feature in self.features.columns:
            if num_classes == 2:
                # Perform t-test for binary classification
                p_value = stats.ttest_ind(self.features[feature][targets[0]],
                                          self.features[feature][targets[1]]).pvalue
            else:
                # Perform ANOVA for multi-class classification
                groups = [self.features[feature][targets == val] for val in set(targets)]
                p_value = stats.f_oneway(*groups).pvalue
            p_values.append(p_value)

        # Construct correlation matrix
        corr_matrix = self.features.corr().abs()

        # Iteratively remove features based on correlation and p-values
        while True:
            # Find pairs of features where correlation exceeds the threshold
            correlated_features = np.where((corr_matrix > corr_threshold) & (corr_matrix < 1))

            if len(correlated_features[0]) == 0:
                break  # Exit if no highly correlated pairs are left

            removal_candidates = []
            for i in range(len(correlated_features[0])):
                idx1, idx2 = correlated_features[0][i], correlated_features[1][i]
                feature1, feature2 = self.features.columns[idx1], self.features.columns[idx2]

                # Keep the feature with the lower p-value (higher statistical significance)
                if p_values[idx1] < p_values[idx2]:
                    removal_candidates.append(feature2)
                else:
                    removal_candidates.append(feature1)

            # Remove duplicated features in the list
            removal_candidates = list(set(removal_candidates))

            # Update the DataFrame and correlation matrix
            initial_number_columns= len(self.features.columns)
            self.features.drop(columns=removal_candidates, inplace=True)
            corr_matrix = self.features.corr().abs()

        print(f"Reduced features from {initial_number_columns} to {len(self.features.columns)}.")

    def process_features(self, corr_threshold=0.8):
        """
        Applies a sequence of preprocessing steps to the feature DataFrame.

        :param corr_threshold: The correlation threshold for feature reduction in reduce_features.
        """
        # Remove specific columns
        columns=['duration_seconds', 'flatness_ratio_10000','flatness_ratio_5000', 'flatness_ratio_1000', 'flatness_ratio_500','flatness_ratio_100',]
        self.drop_columns(columns_to_drop=columns)

        # Reduce features based on statistical tests and correlation
        self.reduce_features(self.targets, corr_threshold=corr_threshold)

        self.processed= True

    # SAVE/LOAD

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

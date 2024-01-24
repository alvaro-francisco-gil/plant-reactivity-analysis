import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
import pickle
import copy

from data.signal_dataset import SignalDataset
from features.wav_feature_extractor import WavFeatureExtractor

class FeaturesDataset(Dataset):
    def __init__(self, signal_dataset: SignalDataset, feature_extractor: WavFeatureExtractor):
        """
        Initialize the FeaturesDataset instance.

        :param signal_dataset: An instance of SignalDataset containing signals and features.
        :param feature_extractor: An instance of WavFeatureExtractor to extract variable features.
        """
        # Extract variable features and columns from the WavFeatureExtractor
        variable_features, variable_columns = feature_extractor.extract_features_multiple_waveforms(
            waveforms=signal_dataset.signals
        )

        # Construct DataFrame for variable features
        variable_features_df = pd.DataFrame(variable_features, columns=variable_columns)

        # Resetting the index of the features attribute of signal_dataset in place
        signal_dataset.features.reset_index(drop=True, inplace=True)

        # Concatenate variable features with features from SignalDataset
        self.features = pd.concat([signal_dataset.features, variable_features_df], axis=1)

        # Label columns are the original columns from SignalDataset
        self.label_columns = list(signal_dataset.features.columns)

        # Variable columns are those extracted by WavFeatureExtractor
        self.variable_columns = variable_columns

        # Inherit the target column from SignalDataset only if it's not None
        self.target_column = signal_dataset.target_column if signal_dataset.target_column is not None else None

        # Inherit the standardization method of the signal
        self.standardization = signal_dataset.standardization if signal_dataset.standardization is not None else None


    # Dataset heritage
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
        # Select only the variable features for the given index
        variable_feature = self.features.loc[idx, self.variable_columns].values

        # Get the target value from the target column
        target = self.features.loc[idx, self.target_column]

        # Convert to PyTorch tensors
        feature_tensor = torch.tensor(variable_feature, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.long)  

        return feature_tensor, target_tensor

    # Getters and Setters
    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        assert isinstance(value, pd.DataFrame), "Features must be a pandas DataFrame."
        self._features = value

    @property
    def label_columns(self):
        return self._label_columns

    @label_columns.setter
    def label_columns(self, value):
        assert isinstance(value, list), "Label columns must be a list."
        self._label_columns = value

    @property
    def variable_columns(self):
        return self._variable_columns

    @variable_columns.setter
    def variable_columns(self, value):
        assert isinstance(value, list), "Variable columns must be a list."
        self._variable_columns = value

    @property
    def target_column(self):
        return self.target_column

    @target_column.setter
    def target_column(self, value):
        assert isinstance(value, str), "Target column must be a string."
        self.target_column = value

    
    # Data Handling
    def drop_columns(self, columns_to_drop):
        """
        Drops specified columns from the features DataFrame.

        :param columns_to_drop: List of column names to be dropped.
        """
        assert isinstance(columns_to_drop, list), "columns_to_drop must be a list."
        # Ensure all columns to drop are in the DataFrame
        for col in columns_to_drop:
            assert col in self.features.columns, f"Column '{col}' not found in the features DataFrame."

        # Drop the columns
        self.features.drop(columns=columns_to_drop, inplace=True)

        # Update label_columns and variable_columns lists
        self.label_columns = [col for col in self.label_columns if col not in columns_to_drop]
        self.variable_columns = [col for col in self.variable_columns if col not in columns_to_drop]

    def drop_rows(self, row_indices):
        """
        Drops specified rows from the features DataFrame.

        :param row_indices: List of indices of the rows to be dropped.
        """
        assert isinstance(row_indices, list), "row_indices must be a list."
        # Ensure all indices are valid
        for idx in row_indices:
            assert 0 <= idx < len(self.features), f"Row index {idx} is out of bounds."

        # Drop the rows and reset the index
        self.features.drop(index=row_indices, inplace=True)
        self.features.reset_index(drop=True, inplace=True)

    def remove_nan_columns(self):
        """
        Removes columns from the features DataFrame that contain NaN values.
        """
        nan_columns = self.features.columns[self.features.isna().any()].tolist()
        if nan_columns:
            print(f"Removing columns with NaN values: {nan_columns}")
            self.features.drop(columns=nan_columns, inplace=True)
        else:
            print("No columns with NaN values found.")

    def remove_nan_rows(self):
        """
        Removes rows from the features DataFrame that contain NaN values.
        """
        nan_rows = self.features[self.features.isna().any(axis=1)].index.tolist()
        if nan_rows:
            print(f"Removing rows with NaN values at indices: {nan_rows}")
            self.features.drop(index=nan_rows, inplace=True)
        else:
            print("No rows with NaN values found.")

    def keep_only_specified_variable_columns(self, columns_to_keep):
        """
        Updates the dataset to keep only the specified variable columns by dropping others.

        :param columns_to_keep: List of variable column names to be kept.
        """
        # Ensure that all columns to keep are in the current variable columns
        assert all(column in self.variable_columns for column in columns_to_keep), "All columns to keep must be in the current variable columns."

        # Determine the variable columns that need to be dropped
        columns_to_drop = [col for col in self.variable_columns if col not in columns_to_keep]

        # Update the variable columns list
        self.variable_columns = columns_to_keep

        # Drop the unwanted columns from the features DataFrame
        self.features.drop(columns=columns_to_drop, inplace=True)

    # Data processing
    def normalize_features(self, method='zscore'):
        """
        Normalizes the variable features in the dataset and returns normalization parameters.

        :param method: The method of normalization. Supported methods include 'zscore' for z-score normalization
                       and 'minmax' for min-max normalization.
        :return: A dictionary containing the normalization parameters used.
        """
        # Work only with variable columns that are numeric
        variable_numeric_cols = [col for col in self.variable_columns if col in self.features.select_dtypes(include='number').columns]

        normalization_params = {}

        if method == 'zscore':
            # Z-score normalization
            mean = self.features[variable_numeric_cols].mean()
            std = self.features[variable_numeric_cols].std()
            std.replace(0, 1, inplace=True)  # Replace 0 std with 1 to avoid division by zero
            self.features[variable_numeric_cols] = (self.features[variable_numeric_cols] - mean) / std
            print("Variable features were properly normalized using 'zscore' method.")

            normalization_params = {'mean': mean, 'std': std}
        elif method == 'minmax':
            # Min-max normalization
            min_val = self.features[variable_numeric_cols].min()
            max_val = self.features[variable_numeric_cols].max()
            self.features[variable_numeric_cols] = (self.features[variable_numeric_cols] - min_val) / (max_val - min_val)
            print("Variable features were properly normalized using 'minmax' method.")

            normalization_params = {'min': min_val, 'max': max_val}
        else:
            raise ValueError("Unsupported normalization method. Choose 'zscore' or 'minmax'.")

        return normalization_params

    def apply_normalization(self, normalization_params):
        """
        Applies normalization to the variable features using provided normalization parameters.

        :param normalization_params: A dictionary containing the normalization parameters.
                                     It should have keys 'mean' and 'std' for 'zscore' method,
                                     or 'min' and 'max' for 'minmax' method.
        """
        # Work only with variable columns that are numeric
        variable_numeric_cols = [col for col in self.variable_columns if col in self.features.select_dtypes(include='number').columns]

        if 'mean' in normalization_params and 'std' in normalization_params:
            # Apply z-score normalization
            mean = normalization_params['mean']
            std = normalization_params['std']
            self.features[variable_numeric_cols] = (self.features[variable_numeric_cols] - mean) / std
            print("Applied z-score normalization.")
        elif 'min' in normalization_params and 'max' in normalization_params:
            # Apply min-max normalization
            min_val = normalization_params['min']
            max_val = normalization_params['max']
            self.features[variable_numeric_cols] = (self.features[variable_numeric_cols] - min_val) / (max_val - min_val)
            print("Applied min-max normalization.")
        else:
            raise ValueError("Invalid normalization parameters. Please provide 'mean' and 'std' for z-score, or 'min' and 'max' for min-max.")
        
    def treat_outliers(self, iqr_multiplier=1.5):
        """
        Treats outliers in the dataset by replacing them with the limit value in the direction of the outlier.
        Outliers are determined based on the specified IQR multiplier.

        :param iqr_multiplier: The multiplier for the IQR to determine outliers. Defaults to 1.5.
        """
        # Work only with variable columns that are numeric
        variable_numeric_cols = [col for col in self.variable_columns if col in self.features.select_dtypes(include='number').columns]

        for feature in variable_numeric_cols:
            Q1 = self.features[feature].quantile(0.25)
            Q3 = self.features[feature].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR

            # Skip columns with no variance
            if IQR == 0:
                print(f"Feature '{feature}' has no variance. Skipping outlier treatment.")
                continue

            # Count and treat outliers
            outliers_lower = self.features[feature] < lower_bound
            outliers_upper = self.features[feature] > upper_bound
            outliers_count = outliers_lower.sum() + outliers_upper.sum()
            total_count = len(self.features[feature])
            outlier_percentage = (outliers_count / total_count) * 100

            # Replace outliers with the nearest boundary value
            self.features[feature] = self.features[feature].mask(outliers_lower, lower_bound)
            self.features[feature] = self.features[feature].mask(outliers_upper, upper_bound)

            print(f"Feature '{feature}': {outlier_percentage:.2f}% outliers treated.")

        print(f"Outliers in variable columns have been treated based on the {iqr_multiplier} * IQR criterion.")

    def reduce_features(self, targets, corr_threshold=0.8):
        assert len(targets) == len(self.features), "Length of targets must match the number of samples."
        assert all(isinstance(target, (int, float)) for target in targets), "Targets must be numeric."

        variable_cols = [col for col in self.variable_columns if col in self.features.columns]

        p_values = {}
        num_classes = len(set(targets))
        for feature in variable_cols:
            if num_classes == 2:
                p_value = stats.ttest_ind(
                    self.features[feature][np.array(targets) == 0],
                    self.features[feature][np.array(targets) == 1]
                ).pvalue
            else:
                groups = [self.features[feature][np.array(targets) == val] for val in set(targets)]
                if len(groups) > 1:
                    p_value = stats.f_oneway(*groups).pvalue
                else:
                    p_value = float('inf')  # Assign a high p-value to effectively skip this feature
            p_values[feature] = p_value

        corr_matrix = self.features[variable_cols].corr().abs()

        while True:
            correlated_pairs = np.where((corr_matrix > corr_threshold) & (corr_matrix < 1))
            if not any(correlated_pairs[0]):
                break

            removal_candidates = set()
            for idx1, idx2 in zip(*correlated_pairs):
                feature1, feature2 = variable_cols[idx1], variable_cols[idx2]

                # Check if both features exist in p_values to avoid KeyError
                if p_values.get(feature1, float('inf')) < p_values.get(feature2, float('inf')):
                    removal_candidates.add(feature2)
                else:
                    removal_candidates.add(feature1)

            initial_variable_columns_count = len(variable_cols)
            self.features.drop(columns=list(removal_candidates), inplace=True)
            variable_cols = [col for col in variable_cols if col not in removal_candidates]
            corr_matrix = self.features[variable_cols].corr().abs()

        print(f"Reduced variable features from {initial_variable_columns_count} to {len(variable_cols)}.")
        self.variable_columns = variable_cols

        return variable_cols

    def get_variable_features_loader(self, targets, batch_size=32, shuffle=True):
        """
        Returns a DataLoader for the variable features and targets of the dataset.

        :param batch_size: The size of each batch.
        :param targets: The target values corresponding to each data point.
        :return: DataLoader for variable features and targets.
        """
        assert len(self.features) == len(targets), "Length of features and targets must be the same."

        class VariableFeaturesDataset(Dataset):
            def __init__(self, features, variable_columns, targets):
                self.features = features[variable_columns]
                self.targets = targets

            def __len__(self):
                return len(self.features)

            def __getitem__(self, idx):
                # Convert features and targets to PyTorch tensors
                feature_tensor = torch.tensor(self.features.iloc[idx].values, dtype=torch.float32)
                target_tensor = torch.tensor(self.targets[idx], dtype=torch.long)  # Change to torch.long for classification tasks
                return feature_tensor, target_tensor

        # Create an instance of the inner dataset class with targets
        variable_features_dataset = VariableFeaturesDataset(self.features, self.variable_columns, targets)

        # Create and return the DataLoader
        return DataLoader(variable_features_dataset, batch_size=batch_size, shuffle=shuffle)

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

    # Save and load
    def save(self, file_path):
        """
        Saves the dataset to a file.

        :param file_path: Path to the file where the dataset will be saved.
        """
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, file_path):
        """
        Loads the dataset from a file.

        :param file_path: Path to the file from which the dataset will be loaded.
        :return: Loaded FeaturesDataset instance.
        """
        with open(file_path, 'rb') as file:
            return pickle.load(file)
        
    def save_to_csv(self, file_path):
        """
        Saves the features DataFrame to a CSV file.

        :param file_path: Path to the CSV file where the dataset will be saved.
        """
        self.features.to_csv(file_path, index=False)

    @classmethod
    def load_from_csv(cls, file_path, label_columns, variable_columns, target_column):
        """
        Loads the dataset from a CSV file.

        :param file_path: Path to the CSV file from which the dataset will be loaded.
        :param label_columns: List of column names to be used as label columns.
        :param variable_columns: List of column names to be used as variable columns.
        :param target_column: Name of the column to be used as the target column.
        :return: Loaded FeaturesDataset instance.
        """
        features = pd.read_csv(file_path)
        return cls(features, label_columns, variable_columns, target_column)
    
    def copy(self):
        """
        Creates a deep copy of the instance.

        :return: A deep copy of the instance.
        """
        return copy.deepcopy(self)

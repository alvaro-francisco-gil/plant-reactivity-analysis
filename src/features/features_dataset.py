import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
import pickle

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

        # Concatenate variable features with features from SignalDataset
        self.features = pd.concat([signal_dataset.features, variable_features_df], axis=1)

        # Label columns are the original columns from SignalDataset
        self.label_columns = list(signal_dataset.features.columns)

        # Variable columns are those extracted by WavFeatureExtractor
        self.variable_columns = variable_columns

        # Inherit the target column from SignalDataset only if it's not None
        if signal_dataset.target_column is not None:
            self.target_column = signal_dataset.target_column

        # Inherit the standardization method of the signal
        if signal_dataset.standardization is not None:
            self.signal_standardization= signal_dataset.standardization

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
        return self._target_column

    @target_column.setter
    def target_column(self, value):
        assert isinstance(value, str), "Target column must be a string."
        self._target_column = value

    
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

    # Data processing
    def normalize_features(self, method='zscore'):
        """
        Normalizes the variable features in the dataset.

        :param method: The method of normalization. Supported methods include 'zscore' for z-score normalization
                       and 'minmax' for min-max normalization.
        """
        # Work only with variable columns that are numeric
        variable_numeric_cols = [col for col in self.variable_columns if col in self.features.select_dtypes(include='number').columns]

        if method == 'zscore':
            # Z-score normalization
            mean = self.features[variable_numeric_cols].mean()
            std = self.features[variable_numeric_cols].std()
            std.replace(0, 1, inplace=True)  # Replace 0 std with 1 to avoid division by zero
            self.features[variable_numeric_cols] = (self.features[variable_numeric_cols] - mean) / std
            print("Variable features were properly normalized using 'zscore' method.")
        elif method == 'minmax':
            # Min-max normalization
            min_val = self.features[variable_numeric_cols].min()
            max_val = self.features[variable_numeric_cols].max()
            self.features[variable_numeric_cols] = (self.features[variable_numeric_cols] - min_val) / (max_val - min_val)
            print("Variable features were properly normalized using 'minmax' method.")
        else:
            raise ValueError("Unsupported normalization method. Choose 'zscore' or 'minmax'.")
        
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
        """
        Reduces features in self.variable_columns based on t-test/ANOVA and correlation matrix.

        :param targets: A list of class values corresponding to the targets.
        :param corr_threshold: Threshold for feature correlation. Features with correlation above this threshold
                               will be considered for removal based on their p-values.
        """
        assert len(targets) == len(self.features), "Length of targets must match the number of samples."
        assert all(isinstance(target, (int, float)) for target in targets), "Targets must be numeric."

        # Work only with variable columns
        variable_cols = [col for col in self.variable_columns if col in self.features.columns]

        # Perform t-test or ANOVA and store p-values
        p_values = {}
        num_classes = len(set(targets))
        for feature in variable_cols:
            if num_classes == 2:
                # Perform t-test for binary classification
                p_value = stats.ttest_ind(
                    self.features[feature][np.array(targets) == 0],
                    self.features[feature][np.array(targets) == 1]
                ).pvalue
            else:
                # Perform ANOVA for multi-class classification
                groups = [self.features[feature][np.array(targets) == val] for val in set(targets)]
                p_value = stats.f_oneway(*groups).pvalue
            p_values[feature] = p_value

        # Construct correlation matrix for variable columns only
        corr_matrix = self.features[variable_cols].corr().abs()

        # Iteratively remove features based on correlation and p-values
        while True:
            correlated_pairs = np.where((corr_matrix > corr_threshold) & (corr_matrix < 1))
            if not any(correlated_pairs[0]):
                break  # Exit if no highly correlated pairs are left

            removal_candidates = set()
            for idx1, idx2 in zip(*correlated_pairs):
                feature1, feature2 = variable_cols[idx1], variable_cols[idx2]

                # Keep the feature with the lower p-value (higher statistical significance)
                if p_values[feature1] < p_values[feature2]:
                    removal_candidates.add(feature2)
                else:
                    removal_candidates.add(feature1)

            # Update the DataFrame, variable columns, and correlation matrix
            initial_variable_columns_count = len(variable_cols)
            self.features.drop(columns=list(removal_candidates), inplace=True)
            variable_cols = [col for col in variable_cols if col not in removal_candidates]
            corr_matrix = self.features[variable_cols].corr().abs()

        print(f"Reduced variable features from {initial_variable_columns_count} to {len(variable_cols)}.")
        self.variable_columns = variable_cols  # Update the variable_columns attribute

###################################?


    def split_dataset_in_loaders(self, test_size=0.3, val_size=0.5, random_state=42, batch_size=32):
        """
        Splits the dataset into training, validation, and testing loaders.

        :param test_size: Proportion of the dataset to include in the test split.
        :param val_size: Proportion of the test set to include in the validation set.
        :param random_state: Random state for reproducibility.
        :return: A tuple (train_loader, val_loader, test_loader).
        """
        dataset_size = len(self)
        indices = list(range(dataset_size))
        train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)
        val_indices, test_indices = train_test_split(test_indices, test_size=val_size, random_state=random_state)

        # Custom collate function to select only variable columns
        def custom_collate(batch):
            # Retrieve only the features from variable_columns
            variable_features = [self.features.iloc[idx][self.variable_columns].values for idx, _ in batch]
            targets = [target for _, target in batch]
            
            # Convert to tensors
            variable_features_tensor = torch.tensor(variable_features, dtype=torch.float32)
            targets_tensor = torch.tensor(targets, dtype=torch.long)  # Adjust dtype if necessary

            return variable_features_tensor, targets_tensor

        train_dataset = Subset(self, train_indices)
        val_dataset = Subset(self, val_indices)
        test_dataset = Subset(self, test_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

        return train_loader, val_loader, test_loader

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

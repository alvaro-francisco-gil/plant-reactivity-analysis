import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import copy
from scipy.stats import f_oneway, ttest_ind
from typing import List, Tuple

from PlantReactivityAnalysis.data.signal_dataset import SignalDataset
from PlantReactivityAnalysis.features.wav_feature_extractor import WavFeatureExtractor
from PlantReactivityAnalysis.data import preparation_eurythmy_data as ped


class FeaturesDataset(Dataset):
    def __init__(
        self,
        features: pd.DataFrame = None,
        label_columns: list = None,
        variable_columns: list = None,
        target_column: str = None,
    ):
        """
        Initialize the FeaturesDataset instance.
        """
        self.features = features  # Direct access
        self.label_columns = label_columns
        self.variable_columns = variable_columns
        self.target_column_name = target_column  # Direct access

        # Standardization attribute can be set externally if required
        self.standardization = None

    # Derived properties
    @property
    def objective_features(self):
        """
        Returns a DataFrame with only the objective (variable) columns.
        """
        return self.features[self.variable_columns]

    @property
    def label_features(self):
        """
        Returns a DataFrame with only the label columns.
        """
        return self.features[self.label_columns]

    # Torch Dataset heritage
    def __len__(self):
        """
        Returns the number of samples in the dataset.

        :return: Integer representing the number of samples.
        """
        return len(self.features)

    def __getitem__(self, idx):
        # Extract features and target for the given index
        variable_feature = self.features.loc[idx, self.variable_columns].values
        target = self.features.loc[idx, self.target_column_name]

        # Ensure data is numeric and convert to appropriate types
        variable_feature = np.asarray(variable_feature, dtype=np.float32)
        target = np.asarray(target, dtype=np.int64)  # or np.float32, depending on your target data

        # Convert to PyTorch tensors
        feature_tensor = torch.tensor(variable_feature, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.long)

        return feature_tensor, target_tensor

    # Data Handling
    def add_target_column(self, column_name, target_values):
        """
        Adds a target column to the dataset.

        :param column_name: Name of the new target column.
        :param target_values: List or Series of target values.
        """
        assert isinstance(target_values, (list, pd.Series)), "Target values must be a list or Pandas Series."
        assert len(target_values) == len(self.features), "Length of target_values must match the number of samples."

        # Add the target column to the features DataFrame
        self.features[column_name] = target_values

        # Update the target column name
        self.target_column_name = column_name

        self.variable_columns.append(column_name)

    def keep_only_specified_variable_columns(self, columns_to_keep):
        """
        Updates the dataset to keep only the specified variable columns by dropping others.

        :param columns_to_keep: List of variable column names to be kept.
        """
        # Ensure that all columns to keep are in the current variable columns
        assert all(
            column in self.variable_columns for column in columns_to_keep
        ), "All columns to keep must be in the current variable columns."

        # Determine the variable columns that need to be dropped
        columns_to_drop = [col for col in self.variable_columns if col not in columns_to_keep]

        # Update the variable columns list
        self.variable_columns = columns_to_keep

        # Drop the unwanted columns from the features DataFrame
        self.features.drop(columns=columns_to_drop, inplace=True)

    def keep_only_specified_rows(self, indexes_to_keep):
        """
        Transforms the dataset into a new one keeping only the specified indexes.

        :param indexes_to_keep: List of indexes to keep in the dataset.
        """
        # Calculate the indexes to drop
        all_indexes = set(range(len(self.features)))
        indexes_to_drop = list(all_indexes - set(indexes_to_keep))

        # Drop the unwanted indexes
        self.features.drop(indexes_to_drop, inplace=True)
        self.features.reset_index(drop=True, inplace=True)

    def get_subset(self, indexes):
        """
        Creates a subset of the dataset based on the given indexes and resets the index.

        :param indexes: A list of indexes to include in the subset.
        :return: A new FeaturesDataset instance containing the subset.
        """
        # Create a deep copy of the current instance
        subset_dataset = self.copy()

        # Keep only the rows corresponding to the provided indexes and reset the index
        subset_dataset.features = subset_dataset.features.iloc[indexes].reset_index(drop=True)

        return subset_dataset

    def split_dataset(
        self, split_by_wav: bool, test_size: float = 0.2, val_size: float = 0.2, random_state: int = None
    ):
        if split_by_wav:
            # Split based on wav files
            train_indexes, val_indexes, test_indexes = ped.get_train_val_test_indexes_by_wav()
        else:
            # Split based on proportions
            indexes = range(len(self.features))
            if val_size == 0:
                # Only perform a train-test split
                train_indexes, test_indexes = train_test_split(indexes, test_size=test_size, random_state=random_state)
                val_indexes = []  # No validation indexes
            else:
                # Perform a train-test split and then a train-validation split
                train_indexes, test_indexes = train_test_split(indexes, test_size=test_size, random_state=random_state)
                relative_val_size = val_size / (1 - test_size)
                train_indexes, val_indexes = train_test_split(
                    train_indexes, test_size=relative_val_size, random_state=random_state
                )

        # Create datasets for each split using the indexes
        train_dataset = self.get_subset(train_indexes)
        test_dataset = self.get_subset(test_indexes)
        val_dataset = self.get_subset(val_indexes) if val_size > 0 else None

        return train_dataset, val_dataset, test_dataset

    # Data cleaning
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
        Removes columns from the features DataFrame that contain NaN values,
        including those listed in the variable_columns list.
        """
        # Find columns with NaN values
        nan_columns = self.objective_features.columns[self.objective_features.isna().any()].tolist()

        # Also consider variable_columns for removal if they contain NaN values
        variable_columns_to_remove = [col for col in self.variable_columns if col in nan_columns]

        # Combine lists while maintaining uniqueness
        all_columns_to_remove = list(set(nan_columns + variable_columns_to_remove))
        if all_columns_to_remove:
            print(f"Removing columns with NaN values: {all_columns_to_remove}")
            self.features.drop(columns=all_columns_to_remove, inplace=True)

            # Also update the variable_columns list by removing the columns that were dropped
            self.variable_columns = [col for col in self.variable_columns if col not in all_columns_to_remove]
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

    def prepare_dataset(self, drop_constant: bool, drop_flatness_columns: bool, drop_nan_columns: bool):
        """
        Prepares the dataset by dropping constant value rows and/or specified columns.

        :param drop_constant: If True, drops rows where 'flatness_ratio_100' has a constant value of 1.
        :param drop_flatness: If True, drops specified flatness ratio columns.
        :return: Modified FeaturesDataset instance.
        """
        if drop_constant:
            indexes_constant_value = self.features[self.features["flatness_ratio_10000"] == 1].index.tolist()
            self.drop_rows(indexes_constant_value)

        if drop_flatness_columns:
            columns_to_drop = [
                "duration_seconds",
                "flatness_ratio_10000",
                "flatness_ratio_5000",
                "flatness_ratio_1000",
                "flatness_ratio_500",
                "flatness_ratio_100",
            ]
            self.drop_columns(columns_to_drop=columns_to_drop)

        if drop_nan_columns:
            self.remove_nan_columns()

    # Data processing
    def normalize_features(self, method="zscore"):
        """
        Normalizes the variable features in the dataset and returns normalization parameters.

        :param method: The method of normalization. Supported methods include 'zscore' for z-score normalization
                       and 'minmax' for min-max normalization.
        :return: A dictionary containing the normalization parameters used.
        """
        # Work only with variable columns that are numeric
        variable_numeric_cols = [
            col
            for col in self.variable_columns
            if col in self.features.select_dtypes(include="number").columns and col != self.target_column_name
        ]

        normalization_params = {}

        if method == "zscore":
            # Z-score normalization
            mean = self.features[variable_numeric_cols].mean()
            std = self.features[variable_numeric_cols].std()
            std.replace(0, 1, inplace=True)  # Replace 0 std with 1 to avoid division by zero
            self.features[variable_numeric_cols] = (self.features[variable_numeric_cols] - mean) / std
            print("Variable features were properly normalized using 'zscore' method.")

            normalization_params = {"mean": mean, "std": std}
        elif method == "minmax":
            # Min-max normalization
            min_val = self.features[variable_numeric_cols].min()
            max_val = self.features[variable_numeric_cols].max()
            self.features[variable_numeric_cols] = (self.features[variable_numeric_cols] - min_val) / (
                max_val - min_val
            )
            print("Variable features were properly normalized using 'minmax' method.")

            normalization_params = {"min": min_val, "max": max_val}
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
        variable_numeric_cols = [
            col
            for col in self.variable_columns
            if col in self.features.select_dtypes(include="number").columns and col != self.target_column_name
        ]
        if "mean" in normalization_params and "std" in normalization_params:
            # Apply z-score normalization
            mean = normalization_params["mean"]
            std = normalization_params["std"]
            self.features[variable_numeric_cols] = (self.features[variable_numeric_cols] - mean) / std
            print("Applied z-score normalization.")
        elif "min" in normalization_params and "max" in normalization_params:
            # Apply min-max normalization
            min_val = normalization_params["min"]
            max_val = normalization_params["max"]
            self.features[variable_numeric_cols] = (self.features[variable_numeric_cols] - min_val) / (
                max_val - min_val
            )
            print("Applied min-max normalization.")
        else:
            raise ValueError("Invalid normalization parameters.")

    def replace_outliers_with_bounds(self, iqr_multiplier=1.5):
        """
        Treats outliers in the dataset by replacing them with the limit value in the direction of the outlier.
        Outliers are determined based on the specified IQR multiplier.

        :param iqr_multiplier: The multiplier for the IQR to determine outliers. Defaults to 1.5.
        """
        # Work only with variable columns that are numeric
        variable_numeric_cols = [
            col for col in self.variable_columns if col in self.features.select_dtypes(include="number").columns
        ]

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

    def reduce_features_based_on_target(self, corr_threshold: float = 0.8,
                                        print_test=False) -> Tuple[List[str], pd.DataFrame]:
        """
        Reduces features based on correlation and significance with respect to a discrete target variable.

        Args:
            corr_threshold (float): Threshold for correlation above which features are considered redundant.

        Returns:
            Tuple[List[str], pd.DataFrame]:
            List of variable columns after reduction and a DataFrame with class averages and p-values.
        """
        variable_cols = [col for col in self.variable_columns if col in self.features.columns]
        corr_matrix = self.features[variable_cols].corr().abs()
        feature_stats = pd.DataFrame(columns=self.features[self.target_column_name].unique().tolist() + ['p_value'])

        while True:
            correlated_pairs = np.where((corr_matrix > corr_threshold) & (corr_matrix < 1))
            if not any(correlated_pairs[0]):
                break

            removal_candidates = set()
            for idx1, idx2 in zip(*correlated_pairs):
                feature1, feature2 = variable_cols[idx1], variable_cols[idx2]
                if self.features[self.target_column_name].nunique() == 2:
                    pval1 = ttest_ind(self.features[self.features[self.target_column_name] == 0][feature1],
                                      self.features[self.features[self.target_column_name] == 1][feature1]).pvalue
                    pval2 = ttest_ind(self.features[self.features[self.target_column_name] == 0][feature2],
                                      self.features[self.features[self.target_column_name] == 1][feature2]).pvalue
                else:
                    groups1 = [group[feature1].values for _, group in self.features.groupby(self.target_column_name)]
                    groups2 = [group[feature2].values for _, group in self.features.groupby(self.target_column_name)]
                    pval1 = f_oneway(*groups1).pvalue
                    pval2 = f_oneway(*groups2).pvalue

                if pval1 > pval2:
                    removal_candidates.add(feature2)
                else:
                    removal_candidates.add(feature1)

                # Collect feature stats
                class_averages1 = self.features.groupby(self.target_column_name)[feature1].mean()
                class_averages2 = self.features.groupby(self.target_column_name)[feature2].mean()
                feature_stats.loc[feature1] = class_averages1.tolist() + [pval1]
                feature_stats.loc[feature2] = class_averages2.tolist() + [pval2]

            self.features.drop(columns=list(removal_candidates), inplace=True)
            variable_cols = [col for col in variable_cols if col not in removal_candidates]
            corr_matrix = self.features[variable_cols].corr().abs()

        print(f"Reduced variable features from initial count to {len(variable_cols)}.")
        self.variable_columns = variable_cols

        feature_stats = feature_stats.sort_values(by='p_value', ascending=True)

        if print_test:
            print(feature_stats)

        return variable_cols, feature_stats

    # Data Visualization
    def print_target_distribution(self):
        """
        Prints the counts and percentages of the target column.
        """
        if self.target_column_name not in self.features.columns:
            raise ValueError(f"Target column '{self.target_column_name}' not found in features DataFrame.")

        target_series = self.features[self.target_column_name]
        counts = target_series.value_counts()
        percentages = target_series.value_counts(normalize=True) * 100

        print("Counts and Percentages:")
        for key in counts.index:
            print(f"Class {key}: Count = {counts[key]}, Percentage = {percentages[key]:.2f}%")

    # Save and load
    def save(self, file_path):
        """
        Saves the dataset to a file.

        :param file_path: Path to the file where the dataset will be saved.
        """
        with open(file_path, "wb") as file:
            pickle.dump(self, file)
        print(f"Dataset saved to {file_path}. Shape: {self.features.shape}")

    @classmethod
    def load(cls, file_path):
        """
        Loads the dataset from a file.

        :param file_path: Path to the file from which the dataset will be loaded.
        :return: Loaded FeaturesDataset instance.
        """
        with open(file_path, "rb") as file:
            dataset = pickle.load(file)
        print(f"Dataset loaded from {file_path}. Shape: {dataset.features.shape}")
        return dataset

    def save_to_csv(self, file_path):
        """
        Saves the features DataFrame to a CSV file.

        :param file_path: Path to the CSV file where the dataset will be saved.
        """
        self.features.to_csv(file_path, index=False)
        print(f"Features saved to CSV {file_path}. Shape: {self.features.shape}")

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
        print(f"Features loaded from CSV {file_path}. Shape: {features.shape}")
        return cls(features, label_columns, variable_columns, target_column)

    @classmethod
    def from_signal_dataset(cls, signal_dataset: SignalDataset, feature_extractor: WavFeatureExtractor):
        """
        Class method to initialize FeaturesDataset instance from SignalDataset and WavFeatureExtractor.
        """
        var_features, var_columns = feature_extractor.extract_features_multiple_waveforms(
            waveforms=signal_dataset.signals
        )
        var_features_df = pd.DataFrame(var_features, columns=var_columns)
        signal_dataset.features.reset_index(drop=True, inplace=True)
        features = pd.concat([signal_dataset.features, var_features_df], axis=1)
        label_columns = list(signal_dataset.features.columns)
        target_column = signal_dataset.target_column if signal_dataset.target_column is not None else None

        return cls(features, label_columns, var_columns, target_column)

    def copy(self):
        """
        Creates a deep copy of the instance.

        :return: A deep copy of the instance.
        """
        return copy.deepcopy(self)

    # Handle Eurythmy Data
    def make_final_dataset(self):

        rows_drop = self.objective_features[(self.objective_features['flatness_ratio_1000'] > 0.75) &
                                            (self.objective_features['flatness_ratio_500'] > 0.85) &
                                            (self.objective_features['flatness_ratio_100'] > 0.999)].index.to_list()
        self.prepare_dataset(drop_constant=False, drop_flatness_columns=True, drop_nan_columns=True)
        self.drop_rows(rows_drop)

    def return_subset_given_research_question(self, rq_number):
        """
        Creates a subset of the dataset based on a specific research question.

        :param rq_number: The research question number.
        :return: A new FeaturesDataset instance containing the subset.
        """
        # Get indexes and targets based on the research question
        indexes, targets = ped.get_indexes_and_targets_by_rq(rq_number, self.label_features)

        # Create a deep copy of the current instance
        subset_dataset = self.copy()

        # Apply modifications to the copy
        subset_dataset.keep_only_specified_rows(indexes)
        subset_dataset.add_target_column("target", targets)

        return subset_dataset

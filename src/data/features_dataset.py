from torch.utils.data import Dataset
import pandas as pd


class FeaturesDataset(Dataset):

    def __init__(self, features: list, targets: list, feature_labels: list = None):
        """
        Initialization method for the AudioDataset.

        :param features: The features of the wav data (e.g., MFCCs, spectrograms).
        :param targets: The targets or targets corresponding to each wav sample.
        :param feature_labels: Optional labels for each feature.
        """
        assert len(features) == len(targets), "Features and targets must have the same length"
        self.features = features
        self.targets = targets
        self.feature_labels = feature_labels

        # Check if feature_labels is not None and if each feature list has the same length as feature_labels
        if feature_labels is not None:
            assert all(len(feature) == len(feature_labels) for feature in features), \
                "Each feature list must have the same length as feature_labels"

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.features)

    def __getitem__(self, idx: int):
        """
        Fetch the data sample and its corresponding target at the specified index.

        :param idx: The index of the data sample to retrieve.
        :return: A tuple containing the data sample and its target.
        """
        return self.features[idx], self.targets[idx]
    
    def save_to_csv(self, filepath):
        """
        Saves the dataset to a CSV file.

        :param filepath: The path to the file where the dataset will be saved.
        """
        # Combining features and targets into a DataFrame
        df = pd.DataFrame(self.features)
        df['target'] = self.targets

        # Save the DataFrame to a CSV file
        df.to_csv(filepath, index=False)

    @classmethod
    def load_from_csv(cls, filepath):
        """
        Loads a dataset from a CSV file.

        :param filepath: The path to the CSV file to load the dataset from.
        :return: An instance of WavDataset with the data loaded from the CSV file.
        """
        # Load the CSV file into a DataFrame
        df = pd.read_csv(filepath)

        # Assuming the last column contains the targets
        features = df.iloc[:, :-1].values.tolist()
        targets = df.iloc[:, -1].values.tolist()

        return cls(features, targets)
    
    def get_targets(self):
        """
        Returns the targets of the dataset.
        """
        return self.targets

    def get_features(self):
        """
        Returns the features of the dataset.
        """
        return self.features

    def get_labels(self):
        """
        Returns the labels of the dataset.
        """
        return self.feature_labels
    
    def get_data(self):
        """
        Returns all the signal-target pairs in the dataset.
        """
        return self.features, self.targets
    
    def remove_nan_columns(self):
        """
        Removes columns from the features that contain NaNs and their corresponding feature labels.
        """
        # Convert features to a DataFrame for easier handling
        df = pd.DataFrame(self.features, columns=self.feature_labels)

        # Find columns with NaNs
        cols_with_nan = df.columns[df.isna().any()].tolist()

        # Remove columns with NaNs
        df.drop(columns=cols_with_nan, inplace=True)

        # Update the feature_labels to match the remaining columns
        self.feature_labels = df.columns.tolist()

        # Update the features attribute
        self.features = df.values.tolist()

        if cols_with_nan:
            print(f"Removed columns with NaNs: {cols_with_nan}")
        else:
            print("No columns with NaNs found.")

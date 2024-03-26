import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import copy


class SignalDataset:
    def __init__(self, signals, features, target_column=None, feature_labels=None, sample_rate: int = 10000):
        """
        Initialize the SignalDataset instance.

        :param signals: A list or array of signal data.
        :param features: A DataFrame or a 2D list/array representing features corresponding to the signals.
        :param target_column: The name of the column in 'features' to be used as the target variable. Can be None.
        :param feature_labels: List of column names for the features DataFrame. If None, default names are assigned.
        :param sample_rate: The sample rate of the signals.
        """
        assert len(signals) == len(features), "The length of signals and features must be the same."

        if not isinstance(features, pd.DataFrame):
            if feature_labels is None and isinstance(features, list) and len(features) > 0:
                feature_labels = [f"feature_{i}" for i in range(len(features[0]))]
            features = pd.DataFrame(features, columns=feature_labels)

        features.reset_index(drop=True, inplace=True)

        self.signals = signals
        self.features = features
        self.sample_rate = sample_rate
        self.target_column = target_column

    # Standardization methods
    @staticmethod
    def standardize_wave_peak(waveform):
        """
        Standardizes an audio waveform by removing the DC offset and normalizing its peak.

        :param waveform: An array representing the audio waveform.
        :return: The standardized waveform.
        """

        dc_offset = np.mean(waveform)
        waveform_without_dc = waveform - dc_offset

        normalized_waveform = waveform_without_dc / np.max(np.abs(waveform_without_dc))

        return normalized_waveform

    @staticmethod
    def standardize_wave_zscore(waveform):
        """
        Standardizes a waveform using z-score transformation.

        Parameters:
        - waveform (np.ndarray): The input waveform to be standardized.

        Returns:
        - standardized_waveform (np.ndarray): The standardized waveform using z-scores.
        """

        mean = np.mean(waveform)
        std_dev = np.std(waveform)

        if std_dev != 0:
            waveform = (waveform - mean) / std_dev

        return waveform

    @staticmethod
    def standardize_wave_min_max(waveform):
        """
        Standardizes a waveform using Min-Max scaling.

        Parameters:
        - waveform (np.ndarray): The input waveform to be standardized.

        Returns:
        - standardized_waveform (np.ndarray): The standardized waveform using Min-Max scaling.
        """

        min_val = np.min(waveform)
        max_val = np.max(waveform)

        standardized_waveform = (waveform - min_val) / (max_val - min_val)

        return standardized_waveform

    def standardize_signals(self, method):
        """
        Applies the specified standardization technique to all signals in the dataset.

        :param method: The standardization method to use ('peak', 'zscore', 'min_max').
        """
        standardization_methods = {
            "peak": self.standardize_wave_peak,
            "zscore": self.standardize_wave_zscore,
            "min_max": self.standardize_wave_min_max,
        }

        if method not in standardization_methods:
            raise ValueError("Invalid standardization method. Choose 'peak', 'zscore', or 'min_max'.")

        # Get the chosen standardization function
        standardize_func = standardization_methods[method]

        # Apply the chosen standardization function to each signal
        self.signals = [standardize_func(signal) for signal in self.signals]
        self.standardization = method

    # Visualization
    def display_dataset(self):
        """
        Displays each signal in the dataset with its corresponding features.
        """
        for i, (signal, features_row) in enumerate(zip(self.signals, self.features.iterrows())):
            plt.figure(figsize=(10, 4))
            time_axis = np.arange(len(signal)) / self.sample_rate  # Convert sample indices to time in seconds
            plt.plot(time_axis, signal)

            # Create a title string from the features
            feature_info = ", ".join([f"{col}: {val}" for col, val in zip(self.features.columns, features_row[1])])
            plt.title(f"Features: {feature_info}")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Amplitude")
            plt.show()

    def display_selected_signals(self, indexes):
        """
        Displays signals for given indexes with their corresponding features.

        :param indexes: A list of indexes for the signals to display.
        """
        for i in indexes:
            signal = self.signals[i]
            features_row = self.features.iloc[i]

            plt.figure(figsize=(10, 4))
            time_axis = np.arange(len(signal)) / self.sample_rate
            plt.plot(time_axis, signal)

            feature_info = ", ".join([f"{col}: {val}" for col, val in features_row.items()])
            plt.title(f"Signal {i} - Features: {feature_info}")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Amplitude")
            plt.show()

    # Signal manipulation
    def remove_signals_by_index(self, indexes):
        """
        Remove signals and their corresponding features given their indexes.

        :param indexes: A list of indexes of the signals to be removed.
        """
        # Filter out signals and features by keeping those not in the provided indexes
        self.signals = [signal for i, signal in enumerate(self.signals) if i not in indexes]
        self.features = self.features.drop(indexes).reset_index(drop=True)

    def segment_signals_by_duration(self, segment_duration, segment_column_name="initial_second"):
        """
        Segments each signal into smaller segments of a specified duration and updates the corresponding features,
        including the start of each segment as a new feature. Resets the index of the features DataFrame.

        :param segment_duration: Duration of each segment in seconds.
        """
        new_signals = []
        new_features = []
        num_samples_per_segment = int(self.sample_rate * segment_duration)

        for idx, (signal, features_row) in enumerate(zip(self.signals, self.features.iterrows())):
            for start in range(0, len(signal), num_samples_per_segment):
                end = start + num_samples_per_segment
                if end <= len(signal):
                    segment = signal[start:end]
                    new_signals.append(segment)

                    # Include the corresponding features for each segment
                    segment_features = features_row[1].copy()
                    initial_second = start / self.sample_rate
                    segment_features[segment_column_name] = initial_second
                    new_features.append(segment_features)

        # Update signals and create a new DataFrame with reset index
        self.signals = new_signals
        self.features = pd.DataFrame(new_features).reset_index(drop=True)

    def segment_signals_by_dict(self, id_column, segments_dict, segment_column_name):
        # Initialize new lists for segmented signals and their features
        new_signals = []
        new_features = []

        for idx, signal in enumerate(self.signals):
            measurement_id = self.features.iloc[idx][id_column]
            if measurement_id in segments_dict:
                for segment_name, (start_sec, end_sec) in segments_dict[measurement_id].items():
                    start_sample = int(start_sec * self.sample_rate)
                    end_sample = int(end_sec * self.sample_rate)
                    new_signal = signal[start_sample:end_sample]

                    # Check to ensure the segment is not empty
                    if len(new_signal) > 0:
                        new_signals.append(new_signal)

                        # Copy current features and add the 'segment' column
                        new_feature_row = self.features.iloc[idx].copy()
                        new_feature_row[segment_column_name] = segment_name
                        new_features.append(new_feature_row)

        # Replace the original signals and features with the new segmented ones
        self.signals = new_signals
        self.features = pd.DataFrame(new_features).reset_index(drop=True)

    def remove_constant_signals(self):
        """
        Remove constant signals and their corresponding features.
        """
        non_constant_signals = []
        non_constant_features = []

        for signal, features_row in zip(self.signals, self.features.iterrows()):
            if len(set(signal)) > 1:  # Check if the signal is non-constant
                non_constant_signals.append(signal)
                non_constant_features.append(features_row[1])  # Append the row data

        self.signals = non_constant_signals
        self.features = pd.DataFrame(non_constant_features, columns=self.features.columns)

    def average_signal(self, indexes):
        """
        Calculates the average of signals at the given indexes.

        :param indexes: List of indexes for which to calculate the average signal.
        :return: The average signal as a list or array.
        """
        assert all(idx < len(self.signals) for idx in indexes), "All indexes must be within the range of the dataset."

        # Initialize a variable to sum the signals
        sum_signal = None

        # Sum the signals at the given indexes
        for idx in indexes:
            signal = self.signals[idx]
            if sum_signal is None:
                sum_signal = signal
            else:
                sum_signal = [sum(x) for x in zip(sum_signal, signal)]

        # Calculate the average of the signals
        avg_signal = [x / len(indexes) for x in sum_signal]
        return avg_signal

    def calculate_average_duration(self):
        """
        Calculate the average duration of all signals in the dataset.

        :return: The average duration of the signals in seconds.
        """
        total_duration = 0
        for signal in self.signals:
            duration = len(signal) / self.sample_rate
            total_duration += duration
        average_duration = total_duration / len(self.signals)
        return average_duration

    def resample_signals(self, target_duration: float):
        """
        Stretch or compress the signals to match a given target duration.

        :param target_duration: The target duration in seconds for all signals.
        """
        new_signals = []
        target_samples = int(target_duration * self.sample_rate)

        for signal in self.signals:
            current_duration = len(signal) / self.sample_rate
            current_samples = len(signal)
            new_time = np.linspace(0, current_duration, num=target_samples)
            original_time = np.linspace(0, current_duration, num=current_samples)
            resampled_signal = np.interp(new_time, original_time, signal)
            new_signals.append(resampled_signal)

        self.signals = new_signals

    # Features manipulation
    def add_features(self, new_features, feature_names=None):
        """
        Add new features to the dataset.

        :param new_features: A 2D list or array of new features to add to the dataset,
                            where each inner list/array represents the features for one signal.
        :param feature_names: A list of names for the new features. If None, default names are assigned.
        """
        assert len(new_features) == len(self.features), "The length of new_features must match the existing features."

        # Convert new_features to a DataFrame if it's not already one
        if not isinstance(new_features, pd.DataFrame):
            if feature_names is None:
                feature_names = [f"new_feature_{i}" for i in range(len(new_features[0]))]
            new_features = pd.DataFrame(new_features, columns=feature_names)

        # Add new features to the existing DataFrame
        self.features = pd.concat([self.features, new_features], axis=1)

    def copy(self):
        """
        Creates a deep copy of the instance.

        :return: A deep copy of the instance.
        """
        return copy.deepcopy(self)

    # Save and load
    def save(self, file_path):
        """
        Saves the dataset to a file.

        :param file_path: Path to the file where the dataset will be saved.
        """
        with open(file_path, "wb") as file:
            pickle.dump(self, file)
        print(f"{len(self.signals)} signals have been saved to {file_path}")

    @classmethod
    def load(cls, file_path):
        """
        Loads the dataset from a file.

        :param file_path: Path to the file from which the dataset will be loaded.
        :return: Loaded FeaturesDataset instance.
        """
        with open(file_path, "rb") as file:
            dataset = pickle.load(file)
        print(f"{len(dataset.signals)} signals have been loaded from {file_path}")
        return dataset

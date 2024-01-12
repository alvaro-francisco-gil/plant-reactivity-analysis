from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class SignalDataset(Dataset):
    def __init__(self, signals, labels, sample_rate: int = 10000):
        """
        Initialize the SignalDataset instance.

        :param signals: A list or array of signal data.
        :param labels: A list of labels corresponding to the signals, each element could be a list of labels.
        :param sample_rate: The sample rate of the signals.
        """
        assert len(signals) == len(labels), "The length of signals and labels must be the same."

        # Convert each label set to a tuple
        formatted_labels = [tuple(label) if isinstance(label, list) else (label,) for label in labels]

        self.signals = signals
        self.labels = formatted_labels
        self.sample_rate = sample_rate
        self.standardization = 'None'

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.signals)

    def __getitem__(self, idx):
        """
        Fetch the signal and its corresponding label at the specified index.
        """
        return self.signals[idx], self.labels[idx]
    
    # GETTERS AND SETTERS

    def get_labels(self):
        """
        Returns the labels of the dataset.
        """
        return self.labels
    
    def get_labels_by_indexes(self, indexes):
        """
        Retrieve labels for specified indexes.

        :param indexes: A list of index positions for which to retrieve labels.
        :return: A list of labels corresponding to the specified indexes.
        """
        # Ensure that the requested indexes are within the bounds of the labels list
        if max(indexes, default=-1) < len(self.labels):
            return [self.labels[idx] for idx in indexes if idx < len(self.labels)]
        else:
            print("Warning: Some requested indexes are out of bounds.")
            return [self.labels[idx] for idx in indexes if idx < len(self.labels)]
        

    def get_labels_by_columns(self, columns):
        """
        Retrieve labels for specified columns.

        :param columns: A list of column positions for which to retrieve label elements.
        :return: A list of lists, each containing the label elements for the specified columns.
        """
        extracted_labels = []

        # Iterate over each label
        for label in self.labels:
            # Extract the elements from the label based on the specified columns
            extracted_label = [label[col] for col in columns if col < len(label)]
            extracted_labels.append(extracted_label)

        return extracted_labels


    def get_signals(self):
        """
        Returns the signals of the dataset.
        """
        return self.signals

    def get_data(self):
        """
        Returns all the signal-label pairs in the dataset.
        """
        return self.signals, self.labels

    def get_datapoint_by_index(self, idx):
        """
        Returns the signal-label pair at the specified index.
        """
        if idx < len(self.signals):
            return self.signals[idx], self.labels[idx]
        else:
            return None

    
    
    # STANDARDIZATION METHODS
        
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

        if std_dev!=0:
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
        standardize_func = None
        if method == 'peak':
            standardize_func = self.standardize_wave_peak
        elif method == 'zscore':
            standardize_func = self.standardize_wave_zscore
        elif method == 'min_max':
            standardize_func = self.standardize_wave_min_max
        else:
            raise ValueError("Invalid standardization method. Choose 'peak', 'zscore', or 'min_max'.")

        # Apply the chosen standardization function to each signal
        self.signals = [standardize_func(signal) for signal in self.signals]

        # Update the standardization attribute
        self.standardization = method

    # MANIPULATE DATASET

    def segment_signals(self, segment_duration):
        """
        Segments each signal into smaller segments of a specified duration.

        :param segment_duration: Duration of each segment in seconds.
        """
        new_signals = []
        new_labels = []
        num_samples_per_segment = int(self.sample_rate * segment_duration)

        for signal, label in zip(self.signals, self.labels):
            for start in range(0, len(signal), num_samples_per_segment):
                end = start + num_samples_per_segment
                if end <= len(signal):
                    segment = signal[start:end]
                    new_signals.append(segment)
                    
                    # Create a new tuple by adding the start second of the segment
                    segment_label = label + (start / self.sample_rate,)  # Concatenate tuples
                    new_labels.append(segment_label)

        self.signals = new_signals
        self.labels = new_labels


    def display_dataset(self):
        """
        Displays each signal in the dataset with its corresponding labels.
        """
        for i, (signal, label) in enumerate(zip(self.signals, self.labels)):
            plt.figure()
            time_axis = np.arange(len(signal)) / self.sample_rate  # Convert sample indices to time in seconds
            plt.plot(time_axis, signal)
            plt.title(f'Labels: {label}')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Amplitude')
            plt.show()

    def display_signal(self, index):
        """
        Displays a single signal from the dataset.

        :param index: The index of the signal to display.
        """
        if index < len(self.signals):
            signal = self.signals[index]
            label = self.labels[index]
            plt.figure()
            time_axis = np.arange(len(signal)) / self.sample_rate  # Convert sample indices to time in seconds
            plt.plot(time_axis, signal)
            plt.title(f'Labels: {label}')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Amplitude')
            plt.show()
        else:
            print(f"Index {index} is out of bounds for the dataset.")

    
    def reduce_signals_given_intervals(self, time_intervals):
        """
        Trims the signals in the dataset based on the provided time intervals.

        :param time_intervals: A list of tuples, each containing start and end times in seconds.
        """
        assert len(time_intervals) == len(self.signals), "Time intervals must match the number of signals."

        trimmed_signals = []
        for signal, (start_time, end_time) in zip(self.signals, time_intervals):
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            trimmed_signal = signal[start_sample:end_sample]
            trimmed_signals.append(trimmed_signal)

        self.signals = trimmed_signals

    def add_labels(self, new_labels):
        """
        Add labels to the dataset.

        :param new_labels: A list of labels to add to the dataset. Each element can be a single value or a list.
        """
        assert len(new_labels) == len(self.labels), "The length of new_labels must match the existing labels."

        updated_labels = []
        for i in range(len(self.labels)):
            # Convert current label tuple to a list
            current_label_list = list(self.labels[i])

            # Check if new_labels[i] is a list or a single value
            if isinstance(new_labels[i], list):
                # Extend the list with the elements of new_labels[i]
                current_label_list.extend(new_labels[i])
            elif new_labels[i] is not None:  # Check for None or nan (float('nan'))
                # Append the single value of new_labels[i]
                current_label_list.append(new_labels[i])

            # Convert back to tuple and add to the updated labels list
            updated_labels.append(tuple(current_label_list))

        self.labels = updated_labels

    def reset_labels(self):
        """
        Resets the labels to an empty tuple for each signal in the dataset.
        """
        if len(self.labels) == len(self.signals):
            self.labels = [tuple() for _ in range(len(self.signals))]
        else:
            print("Warning: The number of labels does not match the number of signals. Labels were not reset.")


    def remove_signals_with_nan_labels(self):
        """
        Removes signals that have any label equal to np.nan.
        """
        # Create a new list for signals and labels without nan labels
        filtered_signals = []
        filtered_labels = []

        # Iterate over each signal and its corresponding label
        for signal, label in zip(self.signals, self.labels):
            # Check if any element in the label tuple is np.nan
            if not any(np.isnan(element) if isinstance(element, float) else False for element in label):
                filtered_signals.append(signal)
                filtered_labels.append(label)

        # Update the signals and labels with the filtered lists
        self.signals = filtered_signals
        self.labels = filtered_labels

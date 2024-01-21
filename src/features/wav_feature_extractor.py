import librosa
import numpy as np
from scipy import stats

class WavFeatureExtractor:
    def __init__(self, sample_rate: int = 10000, mfccs: bool = True, temporal: bool = True, statistical: bool = True):
        """
        Initialize the WavFeatureExtractor with default parameters.

        :param sample_rate: Sample rate of the audio.
        :param mfccs: Whether to extract MFCC features.
        :param temporal: Whether to extract temporal features.
        :param statistical: Whether to extract statistical features.
        """
        self.sample_rate = sample_rate
        self.mfccs = mfccs
        self.temporal = temporal
        self.statistical = statistical

    @staticmethod
    def add_feature(name, func, waveform_data, feature_values, feature_labels):
        """
        Adds a calculated feature to the feature lists, handling exceptions.

        :param name: Name of the feature.
        :param func: Function to calculate the feature.
        :param waveform_data: Data to be used in the feature calculation.
        :param feature_values: List to append the feature value.
        :param feature_labels: List to append the feature name.
        """
        try:
            feature_values.append(func(waveform_data))
        except Exception:
            feature_values.append(np.nan)
        feature_labels.append(name)

    # Standardization Methods
        
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
    
    def standardize_waveform(self, waveform, method):
        """
        Standardizes the waveform using the specified method.

        Parameters:
        - waveform (np.ndarray): The input waveform to be standardized.
        - method (str): The standardization method to use ('peak', 'zscore', 'min_max').

        Returns:
        - standardized_waveform (np.ndarray): The standardized waveform.
        """
        if method == 'peak':
            self.standarization = 'peak'
            return self.standardize_wave_peak(waveform)
        elif method == 'zscore':
            self.standarization = 'zscore'
            return self.standardize_wave_zscore(waveform)
        elif method == 'min_max':
            self.standarization = 'min_max'
            return self.standardize_wave_min_max(waveform)
        else:
            raise ValueError("Invalid standardization method. Choose 'peak', 'zscore', or 'min_max'.")
        
    # Statistical Features
        
    @staticmethod    
    def hjorth(X, D=None):
        """
        From: https://github.com/forrestbao/pyeeg/blob/master/pyeeg/hjorth_mobility_complexity.py

        Compute Hjorth mobility and complexity of a time series from either two
        cases below:
            1. X, the time series of type list (default)
            2. D, a first order differential sequence of X (if D is provided,
            recommended to speed up)

        """

        if D is None:
            D = np.diff(X)
            D = D.tolist()

        D.insert(0, X[0])  # pad the first difference
        D = np.array(D)

        n = len(X)

        M2 = float(sum(D ** 2)) / n
        TP = sum(np.array(X) ** 2)
        M4 = 0
        for i in range(1, len(D)):
            M4 += (D[i] - D[i - 1]) ** 2
        M4 = M4 / n

        return np.sqrt(M2 / TP), np.sqrt(
            float(M4) * TP / M2 / M2
        )  # Hjorth Mobility and Complexity

    @staticmethod
    def dfa(X, Ave=None, L=None):
        """
        From: https://github.com/forrestbao/pyeeg/blob/master/pyeeg/detrended_fluctuation_analysis.py

        Compute Detrended Fluctuation Analysis from a time series X and length of
        boxes L.
        """

        X = np.array(X)

        if Ave is None:
            Ave = np.mean(X)

        Y = np.cumsum(X)
        Y -= Ave

        if L is None:
            L = np.floor(len(X) * 1 / (
                2 ** np.array(list(range(4, int(np.log2(len(X))) - 4))))
            )

        F = np.zeros(len(L))  # F(n) of different given box length n

        for i in range(0, len(L)):
            n = int(L[i])                        # for each box length L[i]
            if n == 0:
                print("time series is too short while the box length is too big")
                print("abort")
                exit()
            for j in range(0, len(X), n):  # for each box
                if j + n < len(X):
                    c = list(range(j, j + n))
                    # coordinates of time in the box
                    c = np.vstack([c, np.ones(n)]).T
                    # the value of data in the box
                    y = Y[j:j + n]
                    # add residue in this box
                    F[i] += np.linalg.lstsq(c, y, rcond=None)[1]
            F[i] /= ((len(X) / n) * n)
        F = np.sqrt(F)

        Alpha = np.linalg.lstsq(np.vstack(
            [np.log(L), np.ones(len(L))]
        ).T, np.log(F), rcond=None)[0][0]

        return Alpha

    def extract_statistical_features(self, waveform_data):
        """
        Calculates various statistical features of the waveform data.
        """
        feature_values = []
        feature_labels = []

        # Hjorth parameters
        try:
            hj = self.hjorth(waveform_data)
            feature_values.extend([hj[0], hj[1]])
            feature_labels.extend(['hjorth_mobility', 'hjorth_complexity'])
        except Exception:
            feature_values.extend([np.nan, np.nan])
            feature_labels.extend(['hjorth_mobility', 'hjorth_complexity'])

        # Standard statistical features
        statistical_features = [
            ('mean', np.mean), ('variance', np.var), ('standard_deviation', np.std), 
            ('interquartile_range', stats.iqr), ('skewness', stats.skew), 
            ('kurtosis', stats.kurtosis), ('dfa', self.dfa)
        ]

        for name, func in statistical_features:
            self.add_feature(name, func, waveform_data, feature_values, feature_labels)

        return feature_values, feature_labels
    
    @staticmethod
    def extract_flatness_ratio(array, threshold):
        """
        Calculates the flatness ratio of an array.

        The flatness ratio is defined as the proportion of the array where
        consecutive values remain the same and exceed a specified threshold length.

        :param array: The input array to analyze.
        :param threshold: The minimum length of consecutive values to be considered 'flat'.
        :return: The flatness ratio of the array.

        The function works as follows:
        - It iterates through the array, tracking sequences of identical values.
        - If a sequence length exceeds the threshold, it is added to the total 'flat' length.
        - The ratio of the total 'flat' length to the array's length is returned.
        """

        current_value = None  # Tracks the current value being compared
        current_length = 0    # Tracks the length of the current sequence of identical values
        total_length = 0      # Accumulates the total length of all 'flat' sequences

        for value in array:
            if value == current_value:
                # If the current value matches the previous one, increment the sequence length
                current_length += 1
            else:
                # If the current value is different, reset the sequence
                current_value = value
                # If the previous sequence was long enough, add its length to the total
                if current_length > threshold:
                    total_length += current_length
                current_length = 1  # Start a new sequence

        # Check the last sequence in the array
        if current_length > threshold:
            total_length += current_length

        # Calculate and return the flatness ratio
        return total_length / len(array)

    def extract_temporal_features(self, waveform, flatness_ratio: bool = True):
        """
        Extracts various temporal features from an audio waveform.
        """
        if not np.issubdtype(waveform.dtype, np.floating):
            waveform = waveform.astype(np.float64)

        feature_values = []
        feature_labels = []

        # Call the static method 'add_feature' for each feature
        self.add_feature('zero_crossing_rate', 
                                                lambda x: np.sum(np.diff(np.sign(x)) != 0) / (2 * len(x)), 
                                                waveform, feature_values, feature_labels)
        self.add_feature('root_mean_square_energy', 
                                                lambda x: np.sqrt(np.mean(x ** 2)), 
                                                waveform, feature_values, feature_labels)
        self.add_feature('slope_sign_changes_ratio', 
                                                lambda x: (np.sum(np.diff(np.sign(np.diff(x))) != 0))/len(x), 
                                                waveform, feature_values, feature_labels)
        
        # More temporal features to add
        self.add_feature('duration_seconds', 
                                                lambda x: len(x) / self.sample_rate, 
                                                waveform, feature_values, feature_labels)
    
        # Flatness Ratios
        if flatness_ratio:
            flatness_ratios = [10000, 5000, 1000, 500, 100]
            for ratio in flatness_ratios:
                self.add_feature(f'flatness_ratio_{ratio}', 
                                                        lambda x: self.extract_flatness_ratio(x, ratio), 
                                                        waveform, feature_values, feature_labels)
        

        return feature_values, feature_labels

    def extract_mfcc_features(self, waveform, n_mfcc: int = 13, n_fft: int = 2000, hop_length: int = 500):
        """
        Extracts MFCC features from an audio waveform, computes the average and standard 
        deviation of each MFCC across time, and returns these statistics along with their labels.

        :param waveform: A numpy array representing the audio waveform.
        :param n_mfcc: Number of MFCCs to return.
        :param n_fft: Length of the FFT window.
        :param hop_length: Number of samples between successive frames.
        :return: A tuple containing two elements: a numpy array of the MFCC statistics and a list of corresponding labels.
        """
        # Ensure the waveform is in floating point format for calculations
        if not np.issubdtype(waveform.dtype, np.floating):
            waveform = waveform.astype(np.float64)

        # Extract MFCCs from the waveform
        mfccs = librosa.feature.mfcc(y=waveform, sr=self.sample_rate, n_mfcc=n_mfcc, 
                                     n_fft=n_fft, hop_length=hop_length)

        # Calculate the average and standard deviation of each MFCC
        mfccs_avg = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

        # Generate labels for each MFCC statistic
        avg_labels = [f'mfcc_{i+1}_avg' for i in range(n_mfcc)]
        std_labels = [f'mfcc_{i+1}_std' for i in range(n_mfcc)]

        # Concatenate the averaged and standard deviation features and their labels
        mfccs_features = np.concatenate((mfccs_avg, mfccs_std))
        feature_labels = avg_labels + std_labels

        return mfccs_features, feature_labels

    def extract_features_waveform(self, waveform):
        """
        Extracts all features using the specified feature extraction methods and
        combines them into a single feature list along with their labels.

        :param waveform: A numpy array representing the audio waveform.
        :return: Two lists - one containing all the feature values and another containing all the corresponding feature labels.
        """
        feature_values = []
        feature_labels = []
        
        if self.mfccs:
            # Extract MFCC features
            mfcc_values, mfcc_labels = self.extract_mfcc_features(waveform)
            feature_values.extend(mfcc_values)
            feature_labels.extend(mfcc_labels)

        if self.temporal:
            # Extract temporal features
            temporal_values, temporal_labels = self.extract_temporal_features(waveform)
            feature_values.extend(temporal_values)
            feature_labels.extend(temporal_labels)

        if self.statistical:
            # Extract statistical features
            statistical_values, statistical_labels = self.extract_statistical_features(waveform)
            feature_values.extend(statistical_values)
            feature_labels.extend(statistical_labels)

        return feature_values, feature_labels

    def extract_features_multiple_waveforms(self, waveforms):
        """
        Extracts features from a list of waveforms using specified feature extraction methods.

        :param waveforms: A list of numpy arrays, each representing an audio waveform.
        :param mfccs: Boolean flag to include MFCC features.
        :param temporal: Boolean flag to include temporal features.
        :param statistical: Boolean flag to include statistical features.
        :return: Two lists - one containing all the feature values from all waveforms,
                 and another containing all the corresponding feature labels.
        """
        all_feature_values = []

        for waveform in waveforms:
            feature_values, feature_labels = self.extract_features_waveform(waveform)
            all_feature_values.append(feature_values)

        return all_feature_values, feature_labels

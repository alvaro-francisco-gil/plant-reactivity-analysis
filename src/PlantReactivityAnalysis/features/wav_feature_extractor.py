import librosa
import numpy as np
from pyAudioAnalysis import MidTermFeatures as aF
from scipy import stats


class WavFeatureExtractor:
    def __init__(
        self,
        sample_rate: int = 10000,
        cepstrals: bool = True,
        pyau_mfccs: bool = True,
        temporal: bool = True,
        statistical: bool = True,
        window_size: float = 1,
        hop_length: float = 1,
    ):
        """
        Initialize the WavFeatureExtractor with default parameters.

        :param sample_rate: Sample rate of the audio.
        :param lib_mfccs: Whether to extract MFCC features.
        :param temporal: Whether to extract temporal features.
        :param statistical: Whether to extract statistical features.
        """
        self.sample_rate = sample_rate
        self.temporal = temporal
        self.statistical = statistical
        self.cepstrals = cepstrals
        self.pyau_mfccs = pyau_mfccs
        self.window_size = window_size
        self.hop_length = hop_length

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

        M2 = float(sum(D**2)) / n
        TP = sum(np.array(X) ** 2)
        M4 = 0
        for i in range(1, len(D)):
            M4 += (D[i] - D[i - 1]) ** 2
        M4 = M4 / n

        return np.sqrt(M2 / TP), np.sqrt(float(M4) * TP / M2 / M2)  # Hjorth Mobility and Complexity

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
            L = np.floor(len(X) * 1 / (2 ** np.array(list(range(4, int(np.log2(len(X))) - 4)))))

        F = np.zeros(len(L))  # F(n) of different given box length n

        for i in range(0, len(L)):
            n = int(L[i])  # for each box length L[i]
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
                    y = Y[j: j + n]
                    # add residue in this box
                    F[i] += np.linalg.lstsq(c, y, rcond=None)[1]
            F[i] /= (len(X) / n) * n
        F = np.sqrt(F)

        Alpha = np.linalg.lstsq(np.vstack([np.log(L), np.ones(len(L))]).T, np.log(F), rcond=None)[0][0]

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
            feature_labels.extend(["hjorth_mobility", "hjorth_complexity"])
        except Exception:
            feature_values.extend([np.nan, np.nan])
            feature_labels.extend(["hjorth_mobility", "hjorth_complexity"])

        # Standard statistical features
        statistical_features = [
            ("mean", np.mean),
            ("variance", np.var),
            ("standard_deviation", np.std),
            ("interquartile_range", stats.iqr),
            ("skewness", stats.skew),
            ("kurtosis", stats.kurtosis),
            ("dfa", self.dfa),
        ]

        for name, func in statistical_features:
            self.add_feature(name, func, waveform_data, feature_values, feature_labels)

        return feature_values, feature_labels

    # Temporal Features
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
        current_length = 0  # Tracks the length of the current sequence of identical values
        total_length = 0  # Accumulates the total length of all 'flat' sequences

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

        self.add_feature(
            "zero_crossing_rate",
            lambda x: np.sum(np.diff(np.sign(x)) != 0) / (2 * len(x)),
            waveform,
            feature_values,
            feature_labels,
        )
        self.add_feature(
            "root_mean_square_energy", lambda x: np.sqrt(np.mean(x**2)), waveform, feature_values, feature_labels
        )
        self.add_feature(
            "slope_sign_changes_ratio",
            lambda x: (np.sum(np.diff(np.sign(np.diff(x))) != 0)) / len(x),
            waveform,
            feature_values,
            feature_labels,
        )

        self.add_feature(
            "duration_seconds", lambda x: len(x) / self.sample_rate, waveform, feature_values, feature_labels
        )

        if flatness_ratio:
            flatness_ratios = [10000, 5000, 1000, 500, 100]
            for ratio in flatness_ratios:
                self.add_feature(
                    f"flatness_ratio_{ratio}",
                    lambda x: self.extract_flatness_ratio(x, ratio),
                    waveform,
                    feature_values,
                    feature_labels,
                )

        return feature_values, feature_labels

    # Frequency Features

    def extract_cepstral_features(self, waveform, n_cepstra: int = 13):
        """
        Extracts cepstral features from a waveform, computes the average and standard
        deviation of each cepstral coefficient across time, and returns these statistics
        along with their labels.

        :param waveform: A numpy array representing the waveform.
        :param n_cepstra: Number of cepstral coefficients to return.
        :return: Two elements tuple: a numpy array of the cepstral statistics and a list of corresponding labels.
        """
        # Ensure the waveform is in floating point format for calculations
        if not np.issubdtype(waveform.dtype, np.floating):
            waveform = waveform.astype(np.float64)

        n_fft = round(self.sample_rate * self.window_size)
        hop_length = round(self.sample_rate * self.hop_length)

        # Compute the Short-Time Fourier Transform (STFT)
        stft = np.abs(librosa.stft(y=waveform, n_fft=n_fft, hop_length=hop_length))

        # Add a small constant before taking the logarithm to avoid log(0)
        epsilon = 1e-9
        log_stft = np.log(stft + epsilon)

        # Compute the real cepstrum
        cepstrum = np.fft.ifft(log_stft).real

        # Take the first n_cepstra coefficients (excluding the zeroth cepstrum)
        cepstra = cepstrum[1: n_cepstra + 1, :]

        # Calculate the average and standard deviation of each cepstral coefficient
        cepstra_avg = np.mean(cepstra, axis=1)
        cepstra_std = np.std(cepstra, axis=1)

        # Generate labels for each cepstral coefficient statistic
        avg_labels = [f"cepstra_{i+1}_avg" for i in range(n_cepstra)]
        std_labels = [f"cepstra_{i+1}_std" for i in range(n_cepstra)]

        # Concatenate the averaged and standard deviation features and their labels
        cepstral_features = np.concatenate((cepstra_avg, cepstra_std))
        feature_labels = avg_labels + std_labels

        return cepstral_features, feature_labels

    def extract_librosa_mfcc_features(self, waveform, n_mfcc: int = 13):
        """
        Extracts MFCC features from an audio waveform, computes the average and standard
        deviation of each MFCC across time, and returns these statistics along with their labels.

        :param waveform: A numpy array representing the audio waveform.
        :param n_mfcc: Number of lib_mfccs to return.
        :param n_fft: Length of the FFT window.
        :param hop_length: Number of samples between successive frames.
        :return: Two elements tuple: a numpy array of the MFCC statistics and a list of corresponding labels.
        """
        # Ensure the waveform is in floating point format for calculations
        if not np.issubdtype(waveform.dtype, np.floating):
            waveform = waveform.astype(np.float64)

        n_fft = round(self.sample_rate * self.window_size)
        hop_length = round(self.sample_rate * self.hop_length)

        # Extract lib_mfccs from the waveform
        lib_mfccs = librosa.feature.mfcc(
            y=waveform, sr=self.sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
        )

        # Calculate the average and standard deviation of each MFCC
        mfccs_avg = np.mean(lib_mfccs, axis=1)
        mfccs_std = np.std(lib_mfccs, axis=1)

        # Generate labels for each MFCC statistic
        avg_labels = [f"lib_mfcc_{i+1}_avg" for i in range(n_mfcc)]
        std_labels = [f"lib_mfcc_{i+1}_std" for i in range(n_mfcc)]

        # Concatenate the averaged and standard deviation features and their labels
        mfccs_features = np.concatenate((mfccs_avg, mfccs_std))
        feature_labels = avg_labels + std_labels

        return mfccs_features, feature_labels

    def extract_pyaudio_mfcc_features(self, waveform, st_window_size=0.05, st_hop_length=0.05):
        mt, st, mt_n = aF.mid_feature_extraction(
            waveform,  # The audio signal (time-domain waveform)
            self.sample_rate,  # Sample rate of the audio signal (in Hz)
            round(self.sample_rate * self.window_size),  # Mid-term window size (in samples)
            round(self.sample_rate * self.hop_length),  # Mid-term window step (in samples)
            round(self.sample_rate * st_window_size),  # Short-term window size (in samples)
            round(self.sample_rate * st_hop_length),  # Short-term window step (in samples),
        )

        # Transpose the mid-term features matrix to have features as rows and windows as columns
        mt_transposed = np.transpose(mt)

        # Perform long-term averaging of mid-term features to get a single feature vector
        mid_term_features_avg = mt_transposed.mean(axis=0)

        return mid_term_features_avg, mt_n

    # Feature Extraction
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

    def extract_features_waveform(self, waveform):
        """
        Extracts all features using the specified feature extraction methods and
        combines them into a single feature list along with their labels.

        :param waveform: A numpy array representing the audio waveform.
        :return: Two lists - all the feature values and all the corresponding feature labels.
        """
        feature_values = []
        feature_labels = []

        if self.cepstrals:
            # Extract MFCC features
            mfcc_values, mfcc_labels = self.extract_cepstral_features(waveform)
            feature_values.extend(mfcc_values)
            feature_labels.extend(mfcc_labels)

        if self.pyau_mfccs:
            # Extract MFCC features
            py_mfcc_values, py_mfcc_labels = self.extract_pyaudio_mfcc_features(waveform)
            feature_values.extend(py_mfcc_values)
            feature_labels.extend(py_mfcc_labels)

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
        :param lib_mfccs: Boolean flag to include MFCC features.
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

import librosa
import numpy as np

class WavFeatureExtractor:
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the WavFeatureExtractor instance.

        :param sample_rate: Sample rate to use for audio files.
        """
        self.sample_rate = sample_rate

    def extract_mfcc(self, audio, n_mfcc=13, hop_length=512, n_fft=2048):
        """
        Extract MFCC features from an audio signal.

        :param audio: The audio time series.
        :param n_mfcc: Number of MFCCs to return.
        :param hop_length: Number of samples between successive frames.
        :param n_fft: Length of the FFT window.
        :return: MFCC features.
        """
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
        return mfccs

    # Additional methods to extract other features like spectrograms, chroma, etc.
    # For example:
    def extract_spectrogram(self, audio, hop_length=512, n_fft=2048):
        """
        Extract a spectrogram from an audio signal.

        :param audio: The audio time series.
        :param hop_length: Number of samples between successive frames.
        :param n_fft: Length of the FFT window.
        :return: Spectrogram.
        """
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        spectrogram = np.abs(stft)
        return spectrogram

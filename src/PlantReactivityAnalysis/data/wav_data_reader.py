import os
import librosa
from typing import List, Tuple, Dict


class WavDataReader:
    def __init__(self, folder: str = None, filename: str = None, sample_rate: int = 10000, extract_key: bool = True):
        """
        Initialize the ElectricalWaveDataReader instance.

        :param folder: The folder that contains the WAV files.
        :param filename: The path to a single WAV file.
        :param sample_rate: The sample rate to use for audio files.
        """
        if not folder and not filename:
            raise ValueError("Either 'folder' or 'filename' must be provided.")
        if folder and filename:
            raise ValueError("'folder' and 'filename' should not be used together.")

        self.sample_rate = sample_rate
        self.data = {}
        self.extract_key = extract_key

        if folder:
            self.read_wav_files_in_folder(folder)
        elif filename:
            self.read_single_wav_file(filename)

        print(f"Total WAV files read: {len(self.data)}")

    @staticmethod
    def extract_key_from_filename(filename: str) -> int:
        """
        Extracts the unique key (id_measurement) from a filename.

        :param filename: The filename to extract the key from.
        :return: The unique identifier (id_measurement).
        """
        key = int(filename.split("_")[1][1:])
        return key

    def read_wav_files_in_folder(self, folder: str):
        """
        Reads all WAV files in the specified folder and stores their audio data
        along with their filenames.

        :param folder: The folder that contains the WAV files.
        """
        for filename in os.listdir(folder):
            if filename.endswith(".wav"):
                try:
                    file_path = os.path.join(folder, filename)
                    audio, _ = librosa.load(file_path, sr=self.sample_rate)
                    if self.extract_key:
                        filename = self.extract_key_from_filename(filename)
                    self.data[filename] = audio
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

    def read_single_wav_file(self, filename: str):
        """
        Reads a single WAV file and stores its audio data.

        :param filename: The path to the WAV file.
        """
        # Load the audio file
        audio, _ = librosa.load(filename, sr=self.sample_rate)

        # Store the audio data
        if self.extract_key:
            filename = self.extract_key_from_filename(filename)
        self.data[filename] = audio

    def get_data(self) -> Dict:
        """
        Returns the loaded audio data with filenames.

        :return: A dictionary of audio data where keys are filenames.
        """
        return self.data

    def get_values(self) -> List:
        """
        Returns a list of the audio data values.

        :return: A list of audio waveforms.
        """
        return list(self.data.values())

    def get_keys(self) -> List:
        """
        Returns a list of the keys (identifiers) for the audio data.

        :return: A list of keys.
        """
        return list(self.data.keys())

    def get_values_and_keys(self) -> Tuple[List[float], List[int]]:
        """
        Returns a list of the values and a list of the keys.

        :return: A list of values, a list of keys.
        """

        return list(self.data.values()), list(self.data.keys())

    def get_sample_rate(self):
        """
        Returns the sample rate

        :return: An integer representing sample_rate
        """
        return self.sample_rate

    def get_ordered_signals_and_keys(self):
        """
        Returns a list of signals ordered by the keys of self.data.
        """
        ordered_keys = sorted(self.data.keys())
        ordered_signals = [self.data[key] for key in ordered_keys]
        return ordered_signals, ordered_keys

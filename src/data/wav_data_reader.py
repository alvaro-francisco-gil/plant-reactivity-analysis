import os
import librosa

class WavDataReader:
    def __init__(self, folder: str, sample_rate: int = 10000):
        """
        Initialize the ElectricalWaveDataReader instance.

        :param folder: The folder that contains the WAV files.
        :param sample_rate: The sample rate to use for audio files.
        """
        self.folder = folder
        self.sample_rate = sample_rate
        self.audio_data = {}

    @staticmethod
    def extract_key_from_filename(filename: str):
        """
        Extracts a unique key or identifier from a filename.

        :param filename: The filename to extract the key from.
        :return: A unique key or identifier.
        """
        key = filename.split('_')[1][1:]
        return key

    def read_wav_files(self):
        """
        Reads all WAV files in the specified folder and stores their audio data
        along with their filenames.
        """
        # Iterate over all files in the folder
        for filename in os.listdir(self.folder):
            if filename.endswith('.wav'):
                # Construct the full path to the file
                file_path = os.path.join(self.folder, filename)

                # Load the audio file
                audio, _ = librosa.load(file_path, sr=self.sample_rate)

                # Store the audio data with the filename as the key
                key= self.extract_key_from_filename(filename)
                self.audio_data[key] = audio

    def get_wav_data(self):
        """
        Returns the loaded audio data with filenames.

        :return: A dictionary of audio data where keys are filenames.
        """
        return self.audio_data


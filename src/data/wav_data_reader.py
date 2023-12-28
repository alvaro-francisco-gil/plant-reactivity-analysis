import os
import librosa

class WavDataReader:

    def __init__(self, folder: str = None, filename: str = None, sample_rate: int = 10000):
        """
        Initialize the ElectricalWaveDataReader instance.

        :param folder: The folder that contains the WAV files.
        :param filename: The path to a single WAV file.
        :param sample_rate: The sample rate to use for audio files.
        """
        self.sample_rate = sample_rate
        self.audio_data = {}

        if folder:
            self.read_wav_files_in_folder(folder)
        elif filename:
            self.read_single_wav_file(filename)

    @staticmethod
    def extract_key_from_filename(filename: str):
        """
        Extracts the unique key (id_measurement) from a filename.

        :param filename: The filename to extract the key from.
        :return: The unique identifier (id_measurement).
        """
        key = int(filename.split('_')[1][1:])
        return key

    def read_wav_files_in_folder(self, folder: str):
        """
        Reads all WAV files in the specified folder and stores their audio data
        along with their filenames.

        :param folder: The folder that contains the WAV files.
        """
        for filename in os.listdir(folder):
            if filename.endswith('.wav'):
                # Construct the full path to the file
                file_path = os.path.join(folder, filename)

                # Load the audio file
                audio, _ = librosa.load(file_path, sr=self.sample_rate)

                # Store the audio data with the key
                key = self.extract_key_from_filename(filename)
                self.audio_data[key] = audio

    def read_single_wav_file(self, filename: str):
        """
        Reads a single WAV file and stores its audio data.

        :param filename: The path to the WAV file.
        """
        # Load the audio file
        audio, _ = librosa.load(filename, sr=self.sample_rate)

        # Store the audio data
        key = self.extract_key_from_filename(filename)
        self.audio_data[key] = audio

    def get_wav_data(self):
        """
        Returns the loaded audio data with filenames.

        :return: A dictionary of audio data where keys are filenames.
        """
        return self.audio_data
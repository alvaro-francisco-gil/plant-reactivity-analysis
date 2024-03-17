import unittest
from unittest.mock import patch
from PlantReactivityAnalysis.data.wav_data_reader import WavDataReader


class TestWavDataReader(unittest.TestCase):

    @patch('os.listdir')
    @patch('os.path.join', side_effect=lambda x, y: f"{x}/{y}")
    @patch('librosa.load', return_value=([0.1, 0.2, 0.3], 22050))
    def test_read_wav_files_in_folder(self, mock_load, mock_join, mock_listdir):
        mock_listdir.return_value = ['file_id01_1.wav', 'file_id02_2.wav']
        reader = WavDataReader(folder='test_folder')

        self.assertEqual(len(reader.get_data()), 2)
        self.assertTrue(1 in reader.get_data())
        self.assertTrue(2 in reader.get_data())

    def test_extract_key_from_filename(self):
        filename = 'file_id01_123.wav'
        key = WavDataReader.extract_key_from_filename(filename)
        self.assertEqual(key, 123)

    @patch('librosa.load', return_value=([0.1, 0.2, 0.3], 22050))
    def test_read_single_wav_file(self, mock_load):
        reader = WavDataReader(filename='file_id01_123.wav')
        self.assertTrue(123 in reader.get_data())

    def test_get_values_and_keys(self):
        reader = WavDataReader()
        reader.data = {1: [0.1, 0.2, 0.3], 2: [0.4, 0.5, 0.6]}
        values, keys = reader.get_values_and_keys()

        self.assertListEqual(values, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self.assertListEqual(keys, [1, 2])

    def test_get_ordered_signals_and_keys(self):
        reader = WavDataReader()
        reader.data = {2: [0.4, 0.5, 0.6], 1: [0.1, 0.2, 0.3]}
        ordered_signals, ordered_keys = reader.get_ordered_signals_and_keys()

        self.assertListEqual(ordered_signals, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self.assertListEqual(ordered_keys, [1, 2])


if __name__ == '__main__':
    unittest.main()

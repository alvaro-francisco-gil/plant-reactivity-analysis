from PlantReactivityAnalysis.data.wav_data_reader import WavDataReader
from PlantReactivityAnalysis.data.signal_dataset import SignalDataset
from PlantReactivityAnalysis.features.wav_feature_extractor import WavFeatureExtractor
from PlantReactivityAnalysis.features.features_dataset import FeaturesDataset
import PlantReactivityAnalysis.data.preparation_eurythmy_data as ped
import PlantReactivityAnalysis.config as cf
import PlantReactivityAnalysis.models.parameters as p


def create_features_datasets(signals=True, letter_features=True, one_sec_features=True):
    """
    Function to create and process datasets as per the user's script.
    """
    raw_signal_dataset_path = cf.RAW_DATA_DIR / "raw_signal_dataset.pkl"
    norm_signal_dataset_path = cf.INTERIM_DATA_DIR / "norm_signal_dataset.pkl"
    raw_letters_signal_dataset_path = cf.RAW_DATA_DIR / "raw_letters_signal_dataset.pkl"
    norm_letters_signal_dataset_path = cf.INTERIM_DATA_DIR / "norm_letters_signal_dataset.pkl"
    raw_1s_signal_dataset_path = cf.RAW_DATA_DIR / "raw_1s_signal_dataset.pkl"
    norm_1s_signal_dataset_path = cf.INTERIM_DATA_DIR / "norm_1s_signal_dataset.pkl"

    if signals:
        # Initialize the Reader with the folder of wavs
        reader = WavDataReader(folder=cf.WAV_FOLDER, sample_rate=10000)
        signals, ids = reader.get_ordered_signals_and_keys()
        meas_df = ped.return_meas_labels_by_keys(ids)

        # SignalDataset 1: Raw
        signal_dataset = SignalDataset(signals=signals, features=meas_df)
        signal_dataset.save(raw_signal_dataset_path)
        signal_dataset = SignalDataset.load(raw_signal_dataset_path)

        # SignalDataset 2: Normalized
        signal_dataset.standardize_signals("zscore")
        signal_dataset.save(norm_signal_dataset_path)

        # SignalDataset 3: Segmented by Letters (raw)
        signal_dataset = SignalDataset.load(raw_signal_dataset_path)
        letter_dictionary = ped.return_letter_dictionary(indexes=signal_dataset.features["id_measurement"].tolist())
        signal_dataset.segment_signals_by_dict("id_measurement", letter_dictionary, "eurythmy_letter")
        signal_dataset.save(raw_letters_signal_dataset_path)

        # SignalDataset 4: Segmented by Letters (normalized)
        signal_dataset = SignalDataset.load(norm_signal_dataset_path)
        signal_dataset.segment_signals_by_dict("id_measurement", letter_dictionary, "eurythmy_letter")
        signal_dataset.save(norm_letters_signal_dataset_path)

        # SignalDataset 5: Segmented in 1s (raw)
        signal_dataset = SignalDataset.load(raw_signal_dataset_path)
        signal_dataset.segment_signals_by_duration(segment_duration=1)
        _ = ped.add_meas_letters(signal_dataset.features)
        signal_dataset.save(raw_1s_signal_dataset_path)

        # SignalDataset 6: Segmented in 1s (normalized)
        signal_dataset = SignalDataset.load(norm_signal_dataset_path)
        signal_dataset.segment_signals_by_duration(segment_duration=1)
        _ = ped.add_meas_letters(signal_dataset.features)
        signal_dataset.save(norm_1s_signal_dataset_path)

    # Features Dataset by Letters
    if letter_features:
        for ws in p.WINDOW_SIZES:
            for rhl in p.RELATIVE_HOP_LENGTHS:
                # Calculate hop length
                hl = ws * rhl

                # Load Extractor and Signal Datasets
                feature_extractor = WavFeatureExtractor(
                    sample_rate=10000,
                    cepstrals=True,
                    pyau_mfccs=True,
                    temporal=True,
                    statistical=True,
                    window_size=ws,
                    hop_length=hl,
                )
                norm_signal_dataset = SignalDataset.load(norm_letters_signal_dataset_path)
                raw_signal_dataset = SignalDataset.load(raw_letters_signal_dataset_path)

                # Create and Save Norm Feature Dataset using Extractor and Signal Dataset
                feat_dataset = FeaturesDataset.from_signal_dataset(norm_signal_dataset, feature_extractor)
                file_name = "features_dataset_norm_letters_ws" + str(ws) + "_hl" + str(hl) + ".pkl"
                feat_norm_letters_dataset_path = cf.FEATURES_LETTERS_DIR / file_name
                feat_dataset.save(feat_norm_letters_dataset_path)

                # Create and Save Raw Feature Dataset using Extractor and Signal Dataset
                feat_dataset = FeaturesDataset.from_signal_dataset(raw_signal_dataset, feature_extractor)
                del raw_signal_dataset
                file_name = "features_dataset_raw_letters_ws" + str(ws) + "_hl" + str(hl) + ".pkl"
                feat_raw_letters_dataset_path = cf.FEATURES_LETTERS_DIR / file_name
                feat_dataset.save(feat_raw_letters_dataset_path)
                del feat_dataset

    # Features Dataset by 1s
    if one_sec_features:
        for ws in p.ONE_SEC_WINDOW_SIZES:
            for rhl in p.RELATIVE_HOP_LENGTHS:
                # Calculate hop length
                hl = ws * rhl

                # Load Extractor and Signal Datasets
                feature_extractor = WavFeatureExtractor(
                    sample_rate=10000,
                    cepstrals=True,
                    pyau_mfccs=True,
                    temporal=True,
                    statistical=True,
                    window_size=ws,
                    hop_length=hl,
                )
                norm_1s_signal_dataset = SignalDataset.load(norm_1s_signal_dataset_path)
                raw_1s_signal_dataset = SignalDataset.load(raw_1s_signal_dataset_path)

                # Create and Save Norm Feature Dataset using Extractor and Signal Dataset
                feat_dataset = FeaturesDataset.from_signal_dataset(norm_1s_signal_dataset, feature_extractor)
                file_name = "features_dataset_norm_1s_ws" + str(ws) + "_hl" + str(hl) + ".pkl"
                feat_norm_1s_dataset_path = cf.FEATURES_ONE_SEC_DIR / file_name
                feat_dataset.save(feat_norm_1s_dataset_path)

                # Create and Save Norm Feature Dataset using Extractor and Signal Dataset
                feat_dataset = FeaturesDataset.from_signal_dataset(raw_1s_signal_dataset, feature_extractor)
                del raw_1s_signal_dataset
                file_name = "features_dataset_raw_1s_ws" + str(ws) + "_hl" + str(hl) + ".pkl"
                feat_raw_1s_dataset_path = cf.FEATURES_ONE_SEC_DIR / file_name
                feat_dataset.save(feat_raw_1s_dataset_path)
                del feat_dataset


if __name__ == "__main__":
    create_features_datasets(signals=True, letter_features=True, one_sec_features=True)

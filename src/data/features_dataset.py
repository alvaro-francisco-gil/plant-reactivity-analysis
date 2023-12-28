from torch.utils.data import Dataset

class WavDataset(Dataset):
    def __init__(self, features: list, labels: list):
        """
        Initialization method for the AudioDataset.

        :param features: The features of the wav data (e.g., MFCCs, spectrograms).
        :param labels: The labels or targets corresponding to each wav sample.
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.features)

    def __getitem__(self, idx: int):
        """
        Fetch the data sample and its corresponding label at the specified index.

        :param idx: The index of the data sample to retrieve.
        :return: A tuple containing the data sample and its label.
        """
        return self.features[idx], self.labels[idx]

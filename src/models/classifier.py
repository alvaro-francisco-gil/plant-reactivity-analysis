"""Implement a features classifier base class"""

from abc import ABC, abstractmethod
from typing import Dict

from sklearn.utils.class_weight import compute_class_weight
import numpy as np


class FeatureClassifier(ABC):
    """
    This class is the base class for all emotion classifiers
    """

    def __init__(
        self,
        name: str = "base",
        features: list = None,
        parameters: Dict = None,
    ) -> None:
        """
        The initializer storing general initialization data

        :param name: The name of the classifier to distinguish
        :param data_type: The data type (text, image, audio, ...)
        :param parameters: Parameter dictionary containing all parameters
        """
        parameters = parameters or {}
        self.name = name
        self.parameters = parameters
        self.features = features
        self.targets = parameters.get("targets")
        self.is_trained = False
        self.logger = None

    @abstractmethod
    def train(self, parameters: Dict, **kwargs) -> None:
        """
        The virtual training method for interfacing

        :param parameters: Parameter dictionary used for training
        :param kwargs: Additional kwargs parameters
        """
        raise NotImplementedError("Abstract class")  # pragma: no cover

    @abstractmethod
    def load(self, parameters: Dict = None, **kwargs) -> None:
        """
        Loading method that loads a previously trained model from disk.

        :param parameters: Parameters required for loading the model
        :param kwargs: Additional kwargs parameters
        """
        raise NotImplementedError("Abstract class")  # pragma: no cover

    @abstractmethod
    def save(self, parameters: Dict = None, **kwargs) -> None:
        """
        Saving method that saves a previously trained model on disk.

        :param parameters: Parameters required for storing the model
        :param kwargs: Additional kwargs parameters
        """
        raise NotImplementedError("Abstract class")  # pragma: no cover

    @abstractmethod
    def classify(self, parameters: Dict, **kwargs) -> np.array:
        """
        The virtual classification method for interfacing

        :param parameters: Parameter dictionary used for classification
        :param kwargs: Additional kwargs parameters
        :return: An array with predicted emotion indices
        """
        raise NotImplementedError("Abstract class")  # pragma: no cover

    def get_class_weights(self):
        """
        Calculates class weights based on the distribution of targets.

        :return: Dictionary containing the class weights
        """
        # Ensure that targets are provided
        if self.targets is None:
            raise ValueError("Targets are not set for the classifier.")

        # Calculate class weights
        class_weights = compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(self.targets), 
            y=self.targets
        )

        # Convert class weights to a dictionary
        class_weights_dict = {class_label: weight for class_label, weight in zip(np.unique(self.targets), class_weights)}

        return class_weights_dict

    @staticmethod
    def init_parameters(parameters: Dict = None, **kwargs) -> Dict:
        """
        Function that merges the parameters and kwargs

        :param parameters: Parameter dictionary
        :param kwargs: Additional parameters in kwargs
        :return: Combined dictionary with parameters
        """
        parameters = parameters or {}
        parameters.update(kwargs)
        return parameters
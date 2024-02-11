import itertools
import pandas as pd


class HyperparameterTuner:
    def __init__(self, model_class, param_grid, train_loader, val_loader, num_epochs, input_size, output_size):
        """
        Initialize the HyperparameterTuner.

        :param model_class: The class of the model to be tuned.
        :param param_grid: Dictionary containing lists of parameters to be tried.
        :param train_loader: DataLoader for the training data.
        :param val_loader: DataLoader for the validation data.
        :param num_epochs: Number of epochs for training.
        :param input_size: The input size for the model.
        :param output_size: The output size for the model.
        """
        self.model_class = model_class
        self.param_grid_keys = list(param_grid.keys())  # Store the keys of the parameter grid
        self.param_grid_values = itertools.product(*param_grid.values())
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.input_size = input_size
        self.output_size = output_size

    def tune(self):
        results = []
        for params in self.param_grid_values:
            # Create a parameter dictionary for hyperparameters
            param_dict = dict(zip(self.param_grid_keys, params))

            # Print the parameters for the current model
            print("Training model with parameters:", param_dict)

            # Instantiate the model with input_size, output_size, and hyperparameters
            model = self.model_class(self.input_size, self.output_size, param_dict)

            # Train the model
            model.train_model(self.train_loader, self.val_loader, self.num_epochs)

            # Check if early stopping was triggered
            if model.early_stop:
                print(f"Early stopping triggered after {model.early_stopping_counter} epochs")

            # Evaluate the model on the validation set
            validation_loss = model.validate(self.val_loader)

            # Store results, including hyperparameters and validation loss
            result_dict = {**param_dict, 'validation_loss': validation_loss}
            results.append(result_dict)

        # Convert results to a DataFrame for easy analysis
        results_df = pd.DataFrame(results)

        # Find the best hyperparameters
        best_hyperparameters = results_df.loc[results_df['validation_loss'].idxmin()]
        return best_hyperparameters, results_df

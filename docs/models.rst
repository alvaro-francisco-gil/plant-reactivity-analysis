Models
======

This section covers the documentation for the modeling aspects of the PlantReactivityAnalysis project. It includes details on the experimentations conducted and the parameters used for the models.

Experiment Module
-----------------
This module is responsible for conducting experiments, which involve training models and evaluating their performance.

.. automodule:: PlantReactivityAnalysis.models.experiment
    :members:
    :undoc-members:
    :show-inheritance:
    :private-members:

Parameter Grid
--------------
The `PARAMETER_GRID` dictionary contains configuration sets for various machine learning models used within the project. Each model's parameters are listed with possible values that can be used during hyperparameter tuning to optimize model performance. Here's how the parameters are organized:

.. code-block:: python

    PARAMETER_GRID = {
        "randomforest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        },
        "gradientboosting": {
            "n_estimators": [100, 200],
            "learning_rate": [0.1, 0.05],
            "max_depth": [3, 5]
        },
        "extratrees": {
            "n_estimators": [100, 200],
            "max_depth": [None, 20],
            "min_samples_split": [2, 5]
        },
        "gaussiannb": {
            "var_smoothing": [1e-09, 1e-08, 1e-10]
        },
        "adaboost": {
            "n_estimators": [50, 100],
            "learning_rate": [0.1, 1.0]
        },
        "kneighbors": {
            "n_neighbors": [5, 10, 15],
            "weights": ["uniform", "distance"]
        },
        "lgbm": {
            "n_estimators": [100, 200],
            "learning_rate": [0.1, 0.05],
            "num_leaves": [31, 64]
        },
        "xgb": {
            "n_estimators": [100, 200],
            "learning_rate": [0.1, 0.05],
            "max_depth": [3, 6]
        }
    }

Model Parameter Descriptions
-----------------------------
Below is a brief description of each parameter listed in the `PARAMETER_GRID`:

- **n_estimators**: The number of trees in the forest or number of boosting stages to perform.
- **max_depth**: The maximum depth of the trees. `None` means that nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
- **min_samples_split**: The minimum number of samples required to split an internal node.
- **learning_rate**: Weight applied to each classifier at each boosting step.
- **var_smoothing**: Portion of the largest variance of all features that is added to variances for calculation stability.
- **n_neighbors**: Number of neighbors to use by default for kneighbors queries.
- **weights**: Weight function used in prediction.
- **num_leaves**: Maximum tree leaves for base learners.
- **max_depth (for xgb/lgbm)**: Maximum depth of a tree.

Each of these parameters can greatly influence the learning algorithm's effectiveness and computational efficiency.

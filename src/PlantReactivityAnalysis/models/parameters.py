# Parameters used in experiments

PARAMETER_GRID = {
    "svm": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    },
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
    "logisticregression": {
        "C": [0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"]
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

PARAMETER_GRID_NO_SCALING = {
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
    "logisticregression": {
        "C": [0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"]
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

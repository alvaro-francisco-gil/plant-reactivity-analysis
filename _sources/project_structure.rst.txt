Project Organization
====================

Here's a detailed explanation of the project's directory and file structure:

.. code-block:: none

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project
    ├── data
    │   ├── experiment     <- Results of the experiment
    │   ├── interim        <- Intermediate data that has been transformed
    │   ├── processed      <- The final, canonical data sets for modeling
    │   └── raw            <- The original, immutable data dump
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   ├── rq1            <- Best models for research question 1
    │   ├── rq2            <- Best models for research question 2
    │   └── rq5            <- Best models for research question 5
    │
    ├── notebooks          <- Exploratory Data Analysis of the project
    │   ├── 1              <- Understand measurements
    │   ├── 2              <- Understand eurythmy letters
    │   ├── 3              <- Display a specific signal
    │   ├── 4              <- Signal error detection (flatness ratio)
    │   ├── 5              <- Average cannonical signals
    │   ├── 6              <- Explore features
    │   └── 7              <- Check experiment results
    │
    ├── pyproject.toml     <- File to store the configuration of black
    │
    ├── reports            <- Images and presentations of the project
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src/               <- Source code for use in this project.
    │   PlantReactivityAnalysis               
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to handle and process data
    │   │   ├── get_data_for_model.py
    │   │   ├── make_dataset.py
    │   │   ├── preparation_eurythmy_data.py
    │   │   ├── signal_dataset.py
    │   │   └── wav_data_reader.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   ├── build_features.py
    │   │   ├── features_dataset.py
    │   │   └── wav_feature_extractor.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make predictions
    │   │   ├── experiment.py
    │   │   └── parameters.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
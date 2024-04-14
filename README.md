# Plant-Reactivity-Analysis
*Research conducted at MiT by Álvaro Francisco Gil*

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/downloads/release/python-31014/)
[![GitHub license](https://badgen.net/github/license/alvaro-francisco-gil/Plant-Reactivity-Analysis)](https://github.com/alvaro-francisco-gil/Plant-Reactivity-Analysis/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/-Docs-green)](https://alvaro-francisco-gil.github.io/Plant-Reactivity-Analysis)
[![Linkedin](https://img.shields.io/badge/-LinkedIn-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/alvaro-francisco-gil/)

The code explores the hypothesis that plants can detect and respond to human movements, especially eurythmy gestures, using their electrical activity. We develop various machine learning models to interpret these signals, treating plants as biosensors for human motion.

## Download the Data
If you want to reproduce the experiment, download the initial dataset from the following link:

[Download dataset](https://www.dropbox.com/scl/fo/sttytnu854wk2lwf19w9c/ADBaVscDYbPDzUjb47bjnYE?rlkey=gxt3w290xw5hnypup9fhrecmv&dl=1)

Place the downloaded data into the `data/raw` directory within the project folder.

## Install the Python Package
Before processing the data, install the project as a Python package using the command below:

```bash
make install
```

## Make Dataset
Transform the raw data into a cleaned, preliminary dataset using the following command:

```bash
    make make_dataset
```

## Build Features
After the initial dataset is prepared, run the feature building script to process the cleaned data into features suitable for modeling:

```bash
    make build_features
```

## Run Experiment
Execute the following command to run the modeling experiments using the prepared features:

```bash
    make experiment
```

## Explore Notebooks
Finally, we recommend running the Jupyter notebooks provided in the project to gain deeper insights into the data and the analysis performed. Start with the introductory notebook and proceed in the order they are numbered:




Project Organization
------------

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


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

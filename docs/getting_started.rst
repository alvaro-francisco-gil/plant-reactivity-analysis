Getting Started
===============

This guide will help you set up the project on a clean installation, detailing steps to obtain the raw data, install necessary components, process the data, and explore the results.

Download the Data
-----------------
First, download the initial dataset from the following link:

- `Download dataset <https://www.dropbox.com/scl/fo/sttytnu854wk2lwf19w9c/ADBaVscDYbPDzUjb47bjnYE?rlkey=gxt3w290xw5hnypup9fhrecmv&dl=1>`_

Place the downloaded data into the ``data/raw`` directory within the project folder.

Install the Python Package
--------------------------
Before processing the data, install the project as a Python package.

.. code-block:: bash

    make install

Make Dataset
------------
Transform the raw data into a cleaned, preliminary dataset using the following command:

.. code-block:: bash

    make make_dataset

This script performs initial data cleaning and transformations required for further processing.

Build Features
--------------
After the initial dataset is prepared, run the feature building script to process the cleaned data into features suitable for modeling:

.. code-block:: bash

    make build_features

This step involves extracting or constructing features from the processed data, which are crucial for effective model training.

Run Experiment
--------------
Execute the following command to run the modeling experiments using the prepared features:

.. code-block:: bash

    make experiment

This will train the models specified in the project and store the results for evaluation.

Explore Notebooks
-----------------
Finally, we recommend running the Jupyter notebooks provided in the project to gain deeper insights into the data and the analysis performed. Start with the introductory notebook and proceed in the order they are numbered:

.. code-block:: bash

    jupyter notebook <notebook_name.ipynb>

These notebooks contain detailed explanations and visualizations of the data and modeling process.

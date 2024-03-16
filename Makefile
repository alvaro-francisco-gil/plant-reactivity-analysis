.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3 make_dataset build_features train_model experiment

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install the Python package
# install:
#	$(PYTHON_INTERPRETER) setup.py develop
install:
	$(PYTHON_INTERPRETER) -m pip install -e .

## Make Dataset
make_dataset:
	$(PYTHON_INTERPRETER) src/PlantReactivityAnalysis/data/make_dataset.py

## Build Features
build_features:
	$(PYTHON_INTERPRETER) src/PlantReactivityAnalysis/features/build_features.py data/processed data/features

## Run Experiment
experiment:
	$(PYTHON_INTERPRETER) src/PlantReactivityAnalysis/models/experiment.py models results

## Train Model
train_model:
	$(PYTHON_INTERPRETER) src/PlantReactivityAnalysis/models/train_model.py data/features models

# Add other existing targets like `clean`, `requirements`, etc.

## Delete all compiled Python files and generated data
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf data/raw/*
	rm -rf data/interim/*
	rm -rf data/processed/*
	rm -rf models/*
	rm -rf results/*

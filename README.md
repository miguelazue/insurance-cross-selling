# Binary Classification of Insurance Cross Selling
Exploration and submission of the competition in the Kaggle Playground Series - Season 4, Episode 7

This repository contains code and notebooks for exploring a dataset and building machine learning models for a binary classification problem related to insurance cross-selling. The project is part of a Kaggle competition where the evaluation metric is the area under the ROC curve (AUC).

## Table of Contents
 - Overview
 - Repository Structure
 - Installation
 - Usage
 - Results
 - Contributing
 - License
 - Acknowledgements


## Overview
In this project, we aim to predict whether a customer will buy an insurance product based on various features. The dataset used for this competition is provided by Kaggle. Our approach involves data exploration, feature engineering, and building various machine learning models to achieve the best performance measured by the AUC.

## Repository Structure
'utility_functions.py': Contains utility functions used throughout the project.
'data_exploration.ipynb': Jupyter notebook for exploring the dataset and performing initial data analysis.
'prediction_models.ipynb': Jupyter notebook for building, training, and evaluating machine learning models.

## Installation
To run the code in this repository, you need to have Python and Jupyter Notebook installed. You can install the required dependencies using pip:

bash
Copy code
pip install -r requirements.txt

## Usage

### Data Exploration:
Open and run the data_exploration.ipynb notebook to explore the dataset and perform initial analysis.

### Prediction Models:
Open and run the prediction_models.ipynb notebook to build, train, and evaluate different machine learning models.
Results

The evaluation metric for this competition is the area under the ROC curve (AUC). Our best model currently achieves an AUC score of 0.8453907317695759.

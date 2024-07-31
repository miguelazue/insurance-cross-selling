# Binary Classification of Insurance Cross Selling

The project is is part of the competition Kaggle Playground Series - Season 4, Episode 7. This repository contains the notebooks for exploring the dataset and building machine learning models for a binary classification problem related to insurance cross-selling. 

## Table of Contents
 - Overview
 - Repository Structure
 - Data Exploration
 - Prediction Models
 - Results

## Overview
The aim of the competition is to predict whether a customer will buy an insurance product based on various features. The dataset used for this competition is provided by Kaggle. The approach involves data exploration, feature engineering, and building various machine learning models to achieve the best performance measured by the AUC.

## Repository Structure

 - `utility_functions.py`: Contains utility functions used throughout the project.
 - `data_exploration.ipynb`: Jupyter notebook for exploring the dataset and performing initial data analysis.
 - `prediction_models.ipynb`: Jupyter notebook for building, training, and evaluating machine learning models.
 - `requirements.txt`: Txt file with the requirements if the code execution wants to be replicated

### Data Exploration Summary:

- Registers with a positive response tend to be older
- Registers with a positive response tend to have a higher annual premium
- There is no clear correlation between age and annual premium.
- Registers with driving license have a higher proportion of responses than registers with no driving license (12.3% vs 5.5%)
- Registers previously insured have a lower proportion of responses than registers previously not insured (0.1% vs 22.8%)
- Registers with an older vehicle have a higher proportion of responses.
- Registers with vehicle damage have a higher proportion of responses than registers with no vehicle damage (24.1% vs 0.4%)
- There is a significant difference between the different region codes and the response proportion. For example for the region code 39 and 44 the proportion of positive response is 0%. On the other hand the Region codes 38 and 28 have a positive response in around 20% of the registers. There are more than 50 differen region codes, a new variable is created identifying the quartiles by proportion of positive responses.
- There is a significant difference between the different sales channels and the response proportion. For example for the region code 27 and 67, there are 0% registers with a positive response. On the other hand the Region codes 123 and 43 have respectively 80% and 70% of positive responses. There are more than 150 differen sales channels, a new variable is created identifying the sextiles by proportion of positive responses.

### Prediction Models and Competition results:

The evaluation metric for this competition is the area under the ROC curve (AUC). The best model achieved an AUC score of 0.85983

- Neural Network keras from Tensorflow - Score:  0.85983
- Random Forest using RandomForestRegressor from sklearn - Score: 0.85880
- Logistic Regression using statsmodels library. - Score: 0.85
- Decision Tree using DecisionTreeRegressor from sklearn - Score: 0.85243


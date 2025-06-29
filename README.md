# Titanic Survival Prediction

This repository contains my solution for the Kaggle Titanic - Machine Learning from Disaster competition. The goal of this challenge is to predict which passengers survived the sinking of the RMS Titanic.

## Overview

This project explores various **feature engineering techniques** and **machine learning models** to predict passenger survival on the Titanic. The current submission achieved a **score of 0.7512** on the Kaggle leaderboard.

I plan to continue iterating on this project, incorporating new ideas for feature engineering and fine-tuning algorithms to improve the prediction accuracy.

## Project Structure

* `titanic_submission.ipynb`: This Jupyter notebook contains all the code for data loading, preprocessing, feature engineering, model training, and prediction.
* `input/`: This directory is expected to contain the `train.csv` and `test.csv` datasets from the Kaggle competition.

## Data Preprocessing and Feature Engineering

The `titanic_submission.ipynb` notebook implements a robust **Scikit-learn Pipeline** to handle data preprocessing and feature engineering steps. Custom transformers were created to encapsulate specific transformations, making the pipeline modular and reproducible.

Key feature engineering steps include:

* **`WasAccompainedTransformer`**: Creates a binary feature indicating whether a passenger was accompanied by family (siblings, spouses, parents, or children).
* **`HadCabinTransformer`**: Extracts a binary feature indicating if a passenger had a recorded cabin, and then drops the original `Cabin` column due to a high number of missing values.
* **`FareOutlierRemover`**: Handles outliers in the `Fare` column by replacing extreme values with the mean of the non-outlier fares.
* **`TitleTransformer`**: Extracts titles from passenger names (e.g., Mr., Miss, Mrs., Master) and groups less common titles into a "Rare" category. The original `Name` column is then dropped.
* **`AgeInputer`**: Imputes missing `Age` values based on the median age associated with their extracted title. This leverages the insight that age distributions often differ significantly across titles (e.g., "Master" typically refers to young boys).
* **`IsChildTransformer`**: Creates a binary feature indicating if a passenger is a minor (age < 18).
* **`FamilySizeTransformer`**: Calculates the total family size for each passenger (`SibSp` + `Parch` + 1).
* **`AgeGroupTransformer`**: (Defined in the notebook but not used in the final pipeline for OneHotEncoding) This transformer creates categorical features for different age groups (Baby, Child, Teenager, Adult, OldAdult, Old).
* **`OneHotTransformer`**: Applies One-Hot Encoding to categorical features such as `Sex`, `Embarked`, and the newly created `Title`.
* **`NaDropper`**: Drops any remaining rows with missing values after imputation.

## Modeling

The solution explores the use of several powerful machine learning models, including:

* **XGBoost Classifier**
* **LightGBM Classifier**
* **CatBoost Classifier**
* **Voting Classifier**: An ensemble method combining the predictions of multiple individual classifiers to leverage their strengths.

The models are evaluated using **cross-validation** to ensure robust performance metrics.

## Results

The current best submission achieved a score of **0.7512** on the Kaggle public leaderboard. This indicates a good starting point, and there's definitely room for further improvement!

## Future Work

* **Advanced Feature Engineering**: Explore more complex feature interactions, such as combining `Pclass` with `Fare` or `Age` with `Sex`.
* **Hyperparameter Tuning**: Conduct more extensive hyperparameter optimization for each model using techniques like GridSearchCV or RandomizedSearchCV.
* **Ensemble Methods**: Experiment with different weighting strategies or stacking ensembles to further improve prediction accuracy.
* **Error Analysis**: Analyze misclassified samples to gain insights into areas where the model struggles and identify potential new features.

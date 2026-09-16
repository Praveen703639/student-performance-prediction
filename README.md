@"
# Student Performance Prediction

End-to-end machine learning project for predicting students' final performance using the UCI Student Performance dataset.

## Overview

This project develops and evaluates regression models to predict the final student grade (`G3`) using demographic, academic, family, social, and lifestyle features.

The project deliberately excludes the intermediate grades `G1` and `G2` to make the prediction task more representative of an early-stage performance prediction setting.

## Dataset

**Source:** UCI Machine Learning Repository  
**Dataset:** Student Performance  
**Dataset ID:** 320

The dataset contains student information from two Portuguese schools.

- 649 student records
- 30 input features
- 1 target variable (`G3`)
- No missing values in the downloaded dataset
- No duplicate records

Dataset: https://archive.ics.uci.edu/dataset/320/student%2Bperformance

Citation:

Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T

## Problem Statement

Student performance is influenced by multiple academic, demographic, family, and social factors. The goal of this project is to build a regression pipeline capable of estimating a student's final grade from available attributes.

## Objectives

- Perform exploratory data analysis
- Prepare numerical and categorical features
- Build a reproducible preprocessing pipeline
- Train multiple regression models
- Compare model performance using MAE, RMSE, and R²
- Perform 5-fold cross-validation
- Tune selected models using GridSearchCV
- Analyze feature importance
- Save trained models and experiment results

## Machine Learning Workflow

```text
Raw Dataset
     |
     v
Data Loading
     |
     v
Train/Test Split
     |
     v
Preprocessing
 ┌───────────────┬────────────────┐
 │ Numerical     │ Categorical    │
 │ Median Impute │ Most Frequent  │
 │ StandardScaler│ One-Hot Encode │
 └───────────────┴────────────────┘
     |
     v
Baseline Models
     |
     v
Test Evaluation
     |
     v
5-Fold Cross-Validation
     |
     v
Hyperparameter Tuning
     |
     v
Tuned Model Evaluation
     |
     v
Feature Importance
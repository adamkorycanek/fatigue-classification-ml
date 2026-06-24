# Fatigue Classification using Biomedical Impedance Data

## Overview

This project implements a machine learning pipeline for classification of fatigue levels based on biomedical impedance measurements collected from multiple subjects.
The dataset was acquired during an experimental biomedical study as part of a Master's thesis in Biomedical Engineering.
A Random Forest classifier is used to classify fatigue levels into three categories: low, medium, and high. The model is evaluated using subject-independent validation.

## Problem Definition

The goal is to predict fatigue level from physiological impedance-based features while ensuring generalization across different subjects.

## Dataset

Real biomedical dataset collected in experimental conditions
Multi-subject structure (subject variability considered)
Features derived from normalized impedance measurements at two frequencies

## Methodology

## Machine Learning Pipeline

Data preprocessing and cleaning
Feature selection and normalization
Leave-One-Subject-Out Cross Validation (LOSO-CV)
Random Forest classification

## Model

Random Forest (scikit-learn)
500 estimators (CV stage)
max_depth=4 (overfitting control)
class_weight="balanced"

## Evaluation

Accuracy score
Classification report (precision, recall, F1-score)
Confusion matrix
Subject-independent validation (LOSO-CV)

## Results

Overall accuracy: 79.2%
Model demonstrates stable performance across subjects

## Key Contributions

End-to-end ML pipeline for biomedical classification
Real experimental dataset (not synthetic)
Subject-independent evaluation strategy (LOSO-CV)
Applied healthcare-oriented machine learning workflow

## Technologies

Python, Scikit-learn, Pandas, NumPy, Matplotlib


## Note

Original system included MATLAB-based measurement hardware integration.
For reproducibility, this repository includes a standalone Streamlit interface using precomputed features.

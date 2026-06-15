# Predictive Maintenance - Engine Failure Prediction: by Leart Ajro

## Overview

- This is a predictive maintenance project that uses machine learning models to predict industrial machine failures using sensor data.

## Problem

- In a real world industrial environment, unplanned machine failures are extremely costly in both time and money. Predictive maintenance uses sensor data to detect failures before they occur, allowing teams to intervene proactively rather than reactively.

## Dataset

- [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020) — obtained from Kaggle

- There are : 
    - 10,000 rows 
    - 5 sensor feature columns
    - 1 target variable
    - 9,661 non failures, 339 failures

## Approach 

- Explored and visualized sensor data to identify patterns and class separation between failure and non-failure classes.

- Preprocessed the data by removing unnecessary columns and handling class imbalance using SMOTE

- Split and scaled the data, then trained and compared multiple models including Logistic Regression and Random Forest.

- Optimized for recall to minimize false negatives, as missed failures carry significant real world cost.

## Results


| Model | Imbalance Strategy | Precision (failure) | Recall (failure) | F1 (failure) | False Negatives |
|---|---|---|---|---|---|
| Logistic Regression | SMOTE | 0.83 | 0.81 | 0.82 | 360 |
| Random Forest | SMOTE | 0.96 | 0.98 | 0.97 | 31 |
| Random Forest | `class_weight='balanced'` | 0.88 | 0.56 | 0.68 | 30 |

- After trying multiple models, and ways of handling the class imbalance. Using SMOTE alongside Random Forest produced the best results.

- This includes a 97% accuracy across the board in precision, recall, and f1 score. 
 
- False negatives went from 360 after using Logistic Regression, to 31 in Random Forest.

- Testing Random Forest with class_weight='balanced' instead of SMOTE revealed that despite 98% overall accuracy, recall on the failure class dropped to 0.56 — demonstrating that accuracy alone is a misleading metric for imbalanced datasets.

## Project Structure

- This project is split into 3 notebooks

    - **Data exploration**, which goes deep into the actual data. Looking at the sensor features, their relationship with both classes, and class separation.

    - **Preprocessing**, this notebook is responsible for cleaning up the data and handling the class imbalance 

    - **modeling**, this is where all the magic happens and we get to see how different models affect the overall accuracy.

## How to Run

1. Clone the repository
2. Create a virtual environment and activate it
3. Install dependencies: `pip install -r requirements.txt`
4. Run notebooks in order: 01 → 02 → 03

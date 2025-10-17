This is the submission for assignment 6 of Kanishq Garg ED22B051 for course DA5401

# DA5401 A6: Imputation via Regression for Missing Data

## Objective

This assignment explores different strategies for handling missing data in a credit risk assessment dataset. The primary goal is to implement and compare four methods for dealing with missing values:

1.  **Median Imputation**
2.  **Linear Regression Imputation**
3.  **Non-Linear Regression Imputation** (using K-Nearest Neighbors)
4.  **Listwise Deletion** (Dropping rows)

The effectiveness of each method is evaluated by training a Logistic Regression classifier on the processed datasets and comparing their predictive performance, with a focus on the F1-score for the positive class (credit default).

## Dataset

The dataset used is the "UCI Credit Card Default Clients Dataset". Missing values were artificially introduced into three numerical columns (`AGE`, `BILL_AMT1`, and `BILL_AMT2`) at a rate of 5% Missing At Random (MAR) to simulate a real-world scenario.

## Methodology

The notebook follows these steps:

1.  **Setup and Environment Configuration**: Import necessary libraries for data manipulation, machine learning, and visualization.
2.  **Data Preprocessing and Imputation**:
    *   Load the original dataset and perform initial cleaning (rename columns, drop 'ID').
    *   Artificially introduce 5% MAR into `AGE`, `BILL_AMT1`, and `BILL_AMT2`.
    *   Create four datasets based on different missing data handling strategies:
        *   **Dataset A (Median Imputation)**: Fill missing values in `AGE`, `BILL_AMT1`, and `BILL_AMT2` with the median of their respective columns.
        *   **Dataset B (Linear Regression Imputation)**: Impute missing `AGE` values using Linear Regression trained on other features, and use median imputation for `BILL_AMT1` and `BILL_AMT2`.
        *   **Dataset C (Non-Linear KNN Imputation)**: Impute missing `AGE` values using K-Nearest Neighbors Regression trained on other features (with hyperparameter tuning for `k`), and use median imputation for `BILL_AMT1` and `BILL_AMT2`.
        *   **Dataset D (Listwise Deletion)**: Drop all rows containing any missing values.
    *   **(Optional Method - IterativeImputer)**: Explore using `IterativeImputer` (MICE) with `BayesianRidge` to impute all missing values iteratively, treating each column with missing values as a target for regression based on all other columns.
3.  **Model Training and Performance Assessment**:
    *   For each of the four datasets (A, B, C, and D), split the data into training and testing sets (80/20 split, stratified by the target variable).
    *   Standardize the features using `StandardScaler` (fitted only on training data).
    *   Train a Logistic Regression classifier on each dataset.
    *   Tune the `C` hyperparameter of the Logistic Regression model using `GridSearchCV` with 5-fold cross-validation, optimizing for the F1-score.
    *   Evaluate the performance of the best model for each dataset on its respective test set using a classification report.
4.  **Comparative Analysis**:
    *   Compile the key performance metrics (Accuracy, Precision, Recall, F1-Score for Class 1) into a summary table.
    *   Visualize the F1-Scores for comparison.
    *   Discuss the trade-offs and efficacy of each imputation strategy based on the results.

## Key Findings

*   **Listwise Deletion** resulted in a significant loss of data and, while achieving high precision, had the lowest recall and F1-score, making it the least effective strategy for this dataset and problem.
*   All **Imputation Methods** (Median, Linear Regression, KNN Regression) preserved the dataset size and outperformed Listwise Deletion in terms of F1-score and recall.
*   **Linear Regression Imputation** and **Non-Linear (KNN) Regression Imputation** for the `AGE` column, combined with median imputation for `BILL_AMT1` and `BILL_AMT2`, performed very similarly.
*   The **Linear Regression Imputation** method achieved the highest F1-score among all strategies tested, indicating the best balance between precision and recall for predicting credit default.
*   The performance similarity between linear and non-linear regression imputation for `AGE` suggests a predominantly linear relationship between `AGE` and the other features in this dataset.
*   The optional **Iterative Imputation (MICE)** method using `BayesianRidge` did not show a significant improvement in F1-score compared to the simpler imputation methods in this case.

## Conclusion and Recommendation

Based on the analysis, **Imputation via Linear Regression** is the recommended strategy for handling missing data in this specific credit risk assessment dataset. It provided the best balance of precision and recall (highest F1-score) while being a more parsimonious model compared to the non-linear KNN imputation. Listwise deletion is not recommended due to the significant loss of valuable data and resulting lower F1-score.

## Requirements

The code in this notebook requires the following libraries:

*   pandas
*   numpy
*   sklearn
*   matplotlib
*   seaborn

These can typically be installed via pip:

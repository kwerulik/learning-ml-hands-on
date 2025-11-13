    # 🏡 Chapter 2: End-to-End Machine Learning Project

This folder contains a complete walkthrough of a Machine Learning project, from fetching the data to evaluating the final model.

## 🎯 Project Goal
The objective is to predict median house values in Californian districts, given a number of features from these districts (e.g., population, median income).

## 🛠️ Workflow & Techniques
I implemented the full ML pipeline:

1.  **Data Acquisition:** Fetching `California Housing Prices` dataset.
2.  **Exploratory Data Analysis (EDA):**
    * Visualizing geographical data.
    * Looking for correlations (Pearson's r).
    * Attribute combinations.
3.  **Data Preparation (Cleaning):**
    * Handling missing values (Imputation).
    * Handling text and categorical attributes (`OneHotEncoder`).
    * Feature Scaling (`StandardScaler`).
    * Building custom Transformation Pipelines.
4.  **Model Training:**
    * Linear Regression.
    * Decision Tree Regressor.
    * Random Forest Regressor.
5.  **Model Evaluation & Fine-Tuning:**
    * Cross-Validation (K-Fold).
    * Grid Search (`GridSearchCV`) to find best hyperparameters.
    * Evaluating on the Test Set (RMSE).

## 📊 Key Libraries Used
* `Pandas` for data manipulation.
* `Matplotlib` for visualization.
* `Scikit-Learn` for pipelines, models, and metrics.
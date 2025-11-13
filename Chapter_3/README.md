# 🔢 Chapter 3: Classification

This folder focuses on classification tasks, moving away from regression. The main playground here is the famous MNIST dataset.

## 🎯 Project Goal

To build a classifier capable of identifying handwritten digits (0-9) and to understand various performance measures.

## 🧠 Key Concepts & Metrics

Accuracy is often not enough for classification, especially with skewed datasets. In this chapter, I implemented and analyzed:

- **Binary Classification:** Detecting if a digit is "5" or "not-5".
- **Multiclass Classification:** Detecting all digits (0-9).
- **Performance Measures:**
  - **Confusion Matrix:** True Positives, False Positives, etc.
  - **Precision & Recall:** The trade-off between them.
  - **F1 Score:** Harmonic mean of precision and recall.
  - **ROC Curve & AUC:** Receiver Operating Characteristic and Area Under Curve.
- **Error Analysis:** Analyzing where the model makes mistakes using confusion matrix visualization.

## 💻 Models Used

- `SGDClassifier` (Stochastic Gradient Descent).
- `RandomForestClassifier`.

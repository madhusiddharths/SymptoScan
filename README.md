# SymptoScan

SymptoScan is a disease classification model designed to distinguish between four closely related diseases: **Cold, Flu, COVID-19, and Allergy**.

## About

A conventional multiclass classifier often fails to classify these diseases accurately because they share many similar features, making a clear decision boundary difficult to establish. 

This project introduces a **Stacking Classifier** approach:
1. **Base Models:** Individual binary classifiers (One-vs-All) are built for each disease.
   - **Allergy:** Decision Tree Classifier
   - **Cold:** Gradient Boosting Classifier
   - **Flu:** Linear SVC
   - **COVID-19:** Random Forest Classifier
2. **Meta-Classifier:** A Support Vector Classifier (SVC) is built on top of the outputs of these base models to make the final prediction.

## Features

- **Multi-model Architecture:** Uses specialized models for different disease profiles.
- **Handling Class Imbalance:** Implements data duplication/oversampling for underrepresented classes (Cold and COVID).
- **Comprehensive Evaluation:** Provides metrics such as Precision, F1-Score, Recall, Accuracy, and Confusion Matrix heatmaps.

## Installation

Ensure you have Python installed. You can install the required dependencies using pip:

```bash
pip install pandas matplotlib seaborn numpy scikit-learn
```

## Usage

1. Prepare your dataset in a file named `disease.csv` (ensure it follows the expected schema with symptoms as features and 'TYPE' as the target).
2. Run the main script:

```bash
python SymptoScan.py
```

## Dataset

The model expects a dataset named `disease.csv` containing symptoms and a `TYPE` column indicating the disease (ALLERGY, COLD, COVID, FLU).

---
*Note: This model is for educational and research purposes. Always consult a healthcare professional for medical diagnosis.*

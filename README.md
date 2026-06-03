# Customer-Churn-Prediction

Overview

Machine Learning solution designed to predict customer churn in a telecommunications business using customer demographics, service usage, and billing information.

The project helps identify customers at risk of leaving, enabling proactive retention strategies and reducing revenue loss.

Business Problem

Customer churn significantly impacts profitability and customer lifetime value.

Organizations need predictive analytics solutions to identify high-risk customers before they discontinue services.

Features

- Data Cleaning
- Feature Engineering
- Logistic Regression Modeling
- Churn Prediction
- Confusion Matrix Evaluation
- Classification Report
- ROC-AUC Analysis
- Churn Driver Identification

Technology Stack

- Python
- Pandas
- NumPy
- Scikit-Learn

Workflow

Customer Dataset
↓
Data Cleaning
↓
Feature Engineering
↓
Model Training
↓
Prediction
↓
Performance Evaluation
↓
Business Insights

Model Metrics

- Accuracy Score
- ROC-AUC Score
- Precision
- Recall
- F1 Score
- 
   Sample Output

==================================================
CUSTOMER CHURN PREDICTION
==================================================

Accuracy: 0.82
ROC-AUC Score: 0.86

Confusion Matrix

[[950 120]
 [140 430]]

Top Churn Drivers

Feature              Coefficient

Contract             1.12
MonthlyCharges       0.84
tenure              -0.76
InternetService      0.65

Key Insights

The model identifies factors such as contract type, monthly charges, tenure, and service usage patterns as significant indicators of customer churn.

Future Enhancements

- Random Forest
- XGBoost
- SHAP Explainability
- Model Deployment using Flask

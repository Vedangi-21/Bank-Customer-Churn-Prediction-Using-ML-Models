# Bank-Customer-Churn-Prediction-Using-ML-Models
Bank Customer Churn Prediction
A machine learning project that predicts customer churn in banking using Random Forest classification, achieving 85.75% accuracy.

<img width="1913" height="912" alt="img1" src="https://github.com/user-attachments/assets/04e8f71f-6f13-4071-8b13-7a29bd0047f9" />
<img width="1183" height="562" alt="img2" src="https://github.com/user-attachments/assets/2edcf027-ccef-4653-8ecd-1c374f3388c8" />
<img width="1888" height="912" alt="img3" src="https://github.com/user-attachments/assets/a99b911f-a94f-444a-83c2-948e8b0f6f8a" />
<img width="1881" height="908" alt="img4" src="https://github.com/user-attachments/assets/53fe6e7f-5d42-488a-ba50-f5e023a469d2" />

📋 Table of Contents

Overview
Dataset
Features
Models Compared
Installation
Usage
Results
Key Insights
Technologies Used
Author


🎯 Overview
Customer churn is a critical challenge for banks. This project uses machine learning to predict which customers are likely to leave, enabling proactive retention strategies. By analyzing customer demographics, financial behavior, and banking patterns, the model identifies at-risk customers with high accuracy.
Problem Statement: Predict whether a bank customer will churn based on their profile and banking behavior.
Business Impact: Early identification of at-risk customers allows banks to implement targeted retention campaigns, potentially saving millions in lost revenue.
📊 Dataset

Source: Kaggle - Bank Customer Churn Prediction
Size: 10,000 customers
Features: 14 attributes including demographics, financial data, and banking behavior
Target Variable: Exited (1 = Churned, 0 = Retained)
Class Distribution: ~20% churn rate

Dataset Features:
FeatureDescriptionTypeCreditScoreCustomer's credit scoreNumericalGeographyCustomer's country (France, Germany, Spain)CategoricalGenderMale/FemaleCategoricalAgeCustomer's ageNumericalTenureYears with the bankNumericalBalanceAccount balanceNumericalNumOfProductsNumber of bank products usedNumericalHasCrCardHas credit card (1/0)BinaryIsActiveMemberActive member status (1/0)BinaryEstimatedSalaryCustomer's salaryNumericalExitedChurned or not (Target)Binary
✨ Features

Exploratory Data Analysis (EDA): Comprehensive visualization and statistical analysis
Data Preprocessing: Handling missing values, outlier detection, feature encoding
Model Training: Three classification algorithms compared
Model Evaluation: Accuracy, precision, recall, F1-score, confusion matrix
Feature Importance Analysis: Identify key churn predictors
Interactive Web UI: Standalone HTML interface for real-time predictions

🤖 Models Compared
ModelAccuracyPrecisionRecallF1-ScoreLogistic Regression78.15%33%5%0.08Decision Tree81.05%55%54%0.55Random Forest85.75%75%47%0.58
Selected Model: Random Forest (best overall performance)
🚀 Installation
Prerequisites

Python 3.8 or higher
pip package manager

Setup

Clone the repository

bashgit clone https://github.com/Vedangi-21/bank-churn-prediction.git
cd bank-churn-prediction

Create a virtual environment 

bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install required packages

bashpip install -r requirements.txt
Requirements.txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
ydata-profiling>=4.0.0

💻 Usage
Running the Jupyter Notebook

Start Jupyter Notebook

bashjupyter notebook

Open Bank_Churn_Prediction.ipynb
Run all cells to:

Load and explore the data
Train models
Generate predictions
View visualizations

Using the Web Interface
Simply open churn_prediction.html in any web browser for an interactive prediction tool.
Making Predictions (Python)
pythonimport pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load trained model (after running notebook)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prepare new customer data
new_customer = {
    'CreditScore': 650,
    'Age': 35,
    'Tenure': 5,
    'Balance': 50000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 100000,
    'Geography_France': 1,
    'Geography_Germany': 0,
    'Geography_Spain': 0,
    'Gender_Female': 0,
    'Gender_Male': 1
}

# Predict
prediction = model.predict([list(new_customer.values())])
probability = model.predict_proba([list(new_customer.values())])

print(f"Churn Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
print(f"Churn Probability: {probability[0][1]*100:.2f}%")
📈 Results
Model Performance

Accuracy: 85.75%
Precision: 75% (of predicted churners, 75% actually churned)
Recall: 47% (model caught 47% of actual churners)

Confusion Matrix
                Predicted
              Stay    Churn
Actual Stay   1520     62
       Churn   222    196
Feature Importance (Top 5)

Age - 100% (most important)
Balance - 78%
Number of Products - 65%
Credit Score - 52%
Geography (Germany) - 45%

💡 Key Insights
From the analysis, we discovered:

Age is the strongest predictor - Older customers (50+) are significantly more likely to churn
Zero balance accounts - Customers with $0 balance have 20% higher churn rate
Product usage matters - Customers with only 1 product are at higher risk
Geographic differences - German customers show higher churn rates than French/Spanish
Inactive members - Non-active members are 2x more likely to leave
Credit score impact - Lower credit scores correlate with higher churn

Business Recommendations:

Target retention campaigns at customers 45+ years old
Incentivize multi-product usage
Re-engage inactive members with personalized offers
Pay special attention to German market dynamics
Monitor accounts with declining balances

🛠️ Technologies Used

Python 3.8+ - Programming language
Pandas - Data manipulation and analysis
NumPy - Numerical computing
Scikit-learn - Machine learning algorithms
Matplotlib & Seaborn - Data visualization
ydata-profiling - Automated EDA
Jupyter Notebook - Interactive development environment
HTML/CSS/JavaScript - Web interface

👤 Author
Vedangi Pagar

GitHub: @Vedangi-21
LinkedIn: linkedin.com/in/vedangipagar
Email: vedangipagar@gmail.com

🙏 Acknowledgments

Dataset provided by Kaggle
Inspired by real-world banking challenges
Thanks to the scikit-learn community for excellent documentation

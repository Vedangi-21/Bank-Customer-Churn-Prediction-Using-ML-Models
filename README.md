# 🏦 Bank Customer Churn Prediction

A machine learning project that predicts customer churn in banking using Random Forest classification, achieving **85.75% accuracy**.

<img width="1913" height="912" alt="img1" src="https://github.com/user-attachments/assets/04e8f71f-6f13-4071-8b13-7a29bd0047f9" />
<img width="1183" height="562" alt="img2" src="https://github.com/user-attachments/assets/2edcf027-ccef-4653-8ecd-1c374f3388c8" />
<img width="1888" height="912" alt="img3" src="https://github.com/user-attachments/assets/a99b911f-a94f-444a-83c2-948e8b0f6f8a" />
<img width="1881" height="908" alt="img4" src="https://github.com/user-attachments/assets/53fe6e7f-5d42-488a-ba50-f5e023a469d2" />
---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Models Compared](#models-compared)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Key Insights](#key-insights)
- [Technologies Used](#technologies-used)
- [Author](#author)

---

## 🎯 Overview

Customer churn is a critical challenge for banks. This project uses machine learning to predict which customers are likely to leave, enabling proactive retention strategies. By analyzing customer demographics, financial behavior, and banking patterns, the model identifies at-risk customers with high accuracy.

**Problem Statement:** Predict whether a bank customer will churn based on their profile and banking behavior.

**Business Impact:** Early identification of at-risk customers allows banks to implement targeted retention campaigns, potentially saving millions in lost revenue.

---

## 📊 Dataset

- **Source:** Kaggle - Bank Customer Churn Prediction
- **Size:** 10,000 customers
- **Features:** 14 attributes including demographics, financial data, and banking behavior
- **Target Variable:** `Exited` (1 = Churned, 0 = Retained)
- **Class Distribution:** ~20% churn rate

### Dataset Features

| Feature | Description | Type |
|---------|-------------|------|
| CreditScore | Customer's credit score | Numerical |
| Geography | Customer's country (France, Germany, Spain) | Categorical |
| Gender | Male/Female | Categorical |
| Age | Customer's age | Numerical |
| Tenure | Years with the bank | Numerical |
| Balance | Account balance | Numerical |
| NumOfProducts | Number of bank products used | Numerical |
| HasCrCard | Has credit card (1/0) | Binary |
| IsActiveMember | Active member status (1/0) | Binary |
| EstimatedSalary | Customer's salary | Numerical |
| Exited | Churned or not (Target) | Binary |

---

## ✨ Features

- **Exploratory Data Analysis (EDA):** Comprehensive visualization and statistical analysis
- **Data Preprocessing:** Handling missing values, outlier detection, feature encoding
- **Model Training:** Three classification algorithms compared
- **Model Evaluation:** Accuracy, precision, recall, F1-score, confusion matrix
- **Feature Importance Analysis:** Identify key churn predictors
- **Interactive Web UI:** Standalone HTML interface for real-time predictions

---

## 🤖 Models Compared

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 78.15% | 33% | 5% | 0.08 |
| Decision Tree | 81.05% | 55% | 54% | 0.55 |
| **Random Forest** | **85.75%** | **75%** | **47%** | **0.58** |

**Selected Model:** Random Forest (best overall performance)

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/Vedangi-21/bank-churn-prediction.git
cd bank-churn-prediction
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**

```bash
pip install -r requirements.txt
```

### Requirements.txt

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
ydata-profiling>=4.0.0
```

---

## 💻 Usage

### Running the Jupyter Notebook

1. Start Jupyter Notebook

```bash
jupyter notebook
```

2. Open `Bank_Churn_Prediction.ipynb`

3. Run all cells to:
   - Load and explore the data
   - Train models
   - Generate predictions
   - View visualizations

### Using the Web Interface

Simply open `churn_prediction.html` in any web browser for an interactive prediction tool.

### Making Predictions (Python)

```python
import pandas as pd
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
```

---

## 📈 Results

### Model Performance

- **Accuracy:** 85.75%
- **Precision:** 75% (of predicted churners, 75% actually churned)
- **Recall:** 47% (model caught 47% of actual churners)

### Confusion Matrix

```
                Predicted
              Stay    Churn
Actual Stay   1520     62
       Churn   222    196
```

### Feature Importance (Top 5)

1. **Age** - 100% (most important)
2. **Balance** - 78%
3. **Number of Products** - 65%
4. **Credit Score** - 52%
5. **Geography (Germany)** - 45%

---

## 💡 Key Insights

From the analysis, we discovered:

1. **Age is the strongest predictor** - Older customers (50+) are significantly more likely to churn
2. **Zero balance accounts** - Customers with $0 balance have 20% higher churn rate
3. **Product usage matters** - Customers with only 1 product are at higher risk
4. **Geographic differences** - German customers show higher churn rates than French/Spanish
5. **Inactive members** - Non-active members are 2x more likely to leave
6. **Credit score impact** - Lower credit scores correlate with higher churn

### Business Recommendations

- Target retention campaigns at customers 45+ years old
- Incentivize multi-product usage
- Re-engage inactive members with personalized offers
- Pay special attention to German market dynamics
- Monitor accounts with declining balances

---

## 🛠️ Technologies Used

- **Python 3.8+** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib & Seaborn** - Data visualization
- **ydata-profiling** - Automated EDA
- **Jupyter Notebook** - Interactive development environment
- **HTML/CSS/JavaScript** - Web interface

---

## 👤 Author

**Vedangi Pagar**

- GitHub: [@Vedangi-21](https://github.com/Vedangi-21)
- LinkedIn: [linkedin.com/in/vedangipagar](https://linkedin.com/in/vedangipagar)
- Email: vedangipagar@gmail.com

---

## 🙏 Acknowledgments

- Dataset provided by Kaggle
- Inspired by real-world banking challenges
- Thanks to the scikit-learn community for excellent documentation

---

⭐ **If you found this project helpful, please give it a star!**

**Last Updated:** January 2026

# Credit Risk Prediction using Machine Learning

## 📌 Project Overview

This project builds a machine learning classification model to predict whether a loan applicant will default on a loan or not.

The objective is to assist financial institutions in identifying high-risk borrowers before approving loans, thereby reducing financial loss.

This is a complete end-to-end ML project including:
- Data preprocessing
- Model training
- Evaluation
- Feature analysis
- Deployment using Streamlit

---

# 🎯 Problem Statement

Given borrower financial and credit-related features, predict:

- 0 → No Default (Safe Borrower)
- 1 → Default (Risky Borrower)

This is a **Binary Classification Problem**.

---

# 🏦 Domain Analysis

In real-world banking, loan default risk depends on:

- Income level
- Loan amount
- Debt burden
- Credit history
- Employment stability
- Previous defaults
- Risk grading (internal credit score)

Banks aim to:
- Minimize False Negatives (missing defaulters)
- Maintain business by reducing False Positives

One key financial metric is:

Loan-to-Income Ratio (loan_percent_income)

Higher loan burden relative to income increases default probability.

---

# 📊 Dataset Description

The dataset contains borrower-level information:

- person_age
- person_income
- person_emp_length
- loan_grade
- loan_amnt
- loan_int_rate
- loan_percent_income
- cb_person_default_on_file
- cb_person_cred_hist_length
- Encoded home ownership
- Encoded loan intent
- loan_status (Target Variable)

Total Records: 32,581  
Target Distribution:
- Safe (0)
- Default (1)

---

# 🔎 Exploratory Data Analysis (EDA)

EDA steps performed:

- Checked dataset structure (`df.info()`)
- Identified missing values
- Checked class balance
- Analyzed feature distributions
- Evaluated feature importance

Key Insights:
- loan_percent_income is the strongest predictor
- Higher loan grades increase risk
- Previous defaults strongly impact prediction
- Lower income increases risk probability

---

# 🧹 Data Preprocessing

1. Handled Missing Values
   - Used median for numerical columns

2. Categorical Encoding
   - Binary mapping for previous default (Y/N → 1/0)
   - Ordinal encoding for loan_grade (A–G → 1–7)
   - One-hot encoding for home ownership and loan intent

3. Train-Test Split
   - 80% training
   - 20% testing

4. Feature Scaling
   - Applied StandardScaler for Logistic Regression
   - Random Forest does not require scaling

---

# 🤖 Models Used

## 1️⃣ Logistic Regression (Baseline Model)

- Linear classifier
- Outputs probability using sigmoid function
- Good for baseline comparison

Performance:
- Accuracy: 84%
- Recall: 49%
- ROC-AUC: 0.86

Limitation:
Missed many defaulters (low recall).

---

## 2️⃣ Random Forest (Final Model)

- Ensemble of decision trees
- Handles non-linear relationships
- Robust to overfitting
- Better at detecting complex patterns

Performance:
- Accuracy: 93%
- Recall: 72%
- ROC-AUC: 0.94

Random Forest was selected as the final model due to higher recall and better discrimination ability.

---

# 📈 Evaluation Metrics Explained

Accuracy:
Overall correctness of predictions.

Precision:
Out of predicted defaulters, how many were actually defaulters.

Recall (Important for Banking):
Out of actual defaulters, how many were correctly identified.

False Negative:
Risky borrower predicted as safe (financial loss).

ROC-AUC:
Measures how well the model separates defaulters from non-defaulters across all thresholds.

AUC = 0.94 indicates excellent class separation.

---

# 🔥 Feature Importance

Top Predictors:

1. loan_percent_income
2. person_income
3. loan_grade
4. loan_int_rate

Key Insight:
Debt burden relative to income is the strongest indicator of default risk.

---

# 💾 Model Saving

The trained Random Forest model was saved using:

joblib.dump(rf_model, "credit_risk_model.pkl")

Scaler was also saved for consistent preprocessing.

This enables deployment without retraining.

---

# 🚀 Deployment (Streamlit App)

A Streamlit web application was developed for real-time prediction.

The app:
- Takes user input (income, loan amount, grade, etc.)
- Loads saved model
- Predicts default probability
- Displays risk classification

To run locally:

streamlit run app.py

---

# 📂 Project Structure
credit-risk-prediction/
│
├── app.py
├── Credit_Risk.ipynb
├── credit_risk_model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
└── credit_risk_dataset.csv

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

---

# 📌 Conclusion

The Random Forest model achieved:

- 93% Accuracy
- 72% Recall
- 0.94 ROC-AUC

The model effectively identifies high-risk borrowers and can assist in credit risk evaluation before loan approval.

This project demonstrates the complete machine learning lifecycle from data analysis to deployment.

Live Demo : https://credit-risk-mlapp.streamlit.app/

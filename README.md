# Credit Scoring Model

## Project Overview
This project is a Machine Learning-based Credit Scoring System that predicts an individual’s creditworthiness using historical financial and demographic data. The system classifies applicants as good credit risk or bad credit risk.

Multiple machine learning models are used and compared to identify the best performing model for credit prediction.

---

## Objective
Predict whether a loan applicant is:

- Good Credit (0)  
- Bad Credit (1)  

using supervised machine learning techniques.

---

## Dataset
The dataset contains financial and personal attributes such as:

- Age  
- Credit amount  
- Account status  
- Employment history  
- Loan purpose  
- Savings status  
- Credit history  
- Number of credits  
- Housing type  
- Job type  

---

## Technologies Used
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

## Machine Learning Models
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  

---

## Project Workflow

- Data Loading  
- Data Exploration (EDA)  
- Data Cleaning  
- Encoding categorical variables  
- Feature-target split  
- Train-test split (stratified)  
- Feature scaling  
- Model training  
- Model evaluation  
- Model comparison  

---

## Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC Score  
- Confusion Matrix  
- ROC Curve  

---

## Results Summary

- Logistic Regression → Strong baseline performance  
- Decision Tree → Moderate performance  
- Random Forest → Best performing model  

Final Model Selected:
Random Forest Classifier

---

## Key Insights

- Credit amount and repayment history are key predictors  
- Financial instability increases default risk  
- Ensemble models perform better for structured financial data  

---

## Business Impact

- Reduces loan default risk  
- Improves credit approval decisions  
- Automates credit scoring process  
- Enhances financial risk management  

---

## Future Improvements

- Hyperparameter tuning (GridSearchCV)  
- Handling class imbalance (SMOTE)  
- Feature selection optimization  
- Model deployment (Flask / Streamlit)  
- Model explainability (SHAP)

---

## Author
Name: Hina
Student ID: CA/SE3/13541
Internship: CodeAlpha Machine Learning Internship  

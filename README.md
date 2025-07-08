# Loan Default Risk Predictor

This project predicts whether a loan applicant is likely to **default** using machine learning. It provides a user-friendly **Streamlit web app** where users can input applicant data and receive a risk prediction instantly.


## 🚀 Live App
[🔗 Click here to try the app](https://loan-default-prediction-nishu.streamlit.app/)  

![](/images/Demo_1.png)
![](/images/Demo_2.pngs)

---
## 📊 Features

- **24 input fields** based on user profile and financial history
- Clean interface with two-column input layout
- Prediction label: `Default` or `No Default`
- Confidence score from logistic regression

---

## 📚 Dataset Overview

Each row in the dataset contains features like:

- `Age`, `Income`, `LoanAmount`, `CreditScore`, `DTIRatio`
- Binary indicators: `HasMortgage`, `HasDependents`, `HasCoSigner`
- Encoded categorical features: `Education`, `EmploymentType`, `MaritalStatus`, `LoanPurpose`

---

## 🧠 Model Training

Three models were trained:

| Model               | ROC-AUC Score |
|--------------------|---------------|
| Logistic Regression| **0.7531** ✅ (Best) |
| Random Forest       | 0.7318        |
| XGBoost             | 0.7425        |

The final deployed model is **Logistic Regression** due to its balance of accuracy, interpretability, and low overfitting.

---

## ⚙️ How to Run Locally

```bash
git clone https://github.com/NishuMehta/Loan-Default-Prediction.git
cd Loan-Default-Prediction
pip install -r requirements.txt
streamlit run app/main.py


## 🙋‍♀️ Author

- [Nishu Mehta](https://github.com/NishuMehta)

## 🔗 Project Link

👉 [GitHub Repository](https://github.com/NishuMehta/Loan-Default-Prediction)
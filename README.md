# Student Performance Prediction

A machine learning project built to predict a student's final academic score based on their study habits, attendance, past academic performance, and assignment completion metrics.

---

## 📌 Project Stages & Workflow

1. **Problem Definition:** Predicting final continuous student scores to help educators and academic advisors identify at-risk students early.
2. **Dataset Collection:** Utilization of a structured student performance dataset containing study hours, attendance percentages, previous scores, and assignment completion rates.
3. **Exploration:** Inspecting distributions, feature correlations, missing values, and data duplicates inside `notebooks/student_analysis.ipynb.
4. **Preparation & Splitting:** Cleaning data, defining features ($X$) and target ($y$), and splitting into training and testing sets.
5. **Training:** Training regression models including Linear Regression, Decision Trees, and Random Forest Regressors.
6. **Evaluation:** Comparing performance using Mean Absolute Error (MAE) and $R^2$ Score on the held-out test set.
7. **Prediction:** Making inferences on new student profiles using standalone scripts.
8. **Model Persistence:** Saving the trained model using **Joblib** and reloading it for production prediction tasks.

---

## 🗂️ Suggested Project Structure

```text
student_ml_project/
├── data/
│   └── student_performance.csv
├── notebooks/
│   └── student_analysis.ipynb
├── models/
│   └── student_model.pkl
├── train.py
├── predict.py
├── requirements.txt
└── README.md
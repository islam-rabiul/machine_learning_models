import joblib
import pandas as pd
model = joblib.load('models/student_model.pkl')
score=pd.DataFrame({
    'Hours Studied': [7],
    'Attendance (%)': [85],
    'Previous Score': [59],
    'Assignment_Completion(%)': [49]
})
predicted_score = model.predict(score)

print("--- New Prediction ---")
print(f"Predicted Final Score: {predicted_score[0]:.2f}")
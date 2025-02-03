from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load your trained model and scaler
model_rf = joblib.load('heart_disease.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/columns-info')
def columns_info():
    return render_template('columns_info.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    user_input = {feature: [float(request.form[feature])] for feature in features}
    user_DF = pd.DataFrame(user_input)

    # Normalize the input using the scaler
    user_normalized = scaler.transform(user_DF)

    # Get prediction probabilities
    probabilities = model_rf.predict_proba(user_normalized)
    prediction = model_rf.predict(user_normalized)

    print("Prediction Probabilities:", probabilities)  # Debugging line
    print("Input Data (Normalized):", user_normalized)

    return render_template('result.html', result=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)

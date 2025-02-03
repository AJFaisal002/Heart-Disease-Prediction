# ❤️ Heart Disease Prediction

A **Flask-based web application** that predicts the likelihood of heart disease using a trained **Random Forest Classifier**. The app takes input parameters such as age, cholesterol level, and blood pressure to provide predictions, helping individuals assess their heart health.

![Heart Disease Prediction](https://img.shields.io/badge/Heart%20Health%20App-Powered%20by%20AI-blue)
![Contributors](https://img.shields.io/badge/Contributors-2-orange)
![Python](https://img.shields.io/badge/Python-3.x-brightgreen)
![Flask](https://img.shields.io/badge/Flask-2.0+-yellow)

---

## 🚀 Features
- Predicts heart disease based on 13 health parameters.
- Interactive web-based interface.
- Powered by a trained **Random Forest Classifier**.
- Includes a preprocessed dataset and trained model.
- Easy-to-use form for user inputs.

---

## 📂 Project Structure
Here’s an overview of the files and folders:

- **`app.py`**: The main Flask application to handle user input and prediction.
- **`heart_disease.pkl`**: The trained machine learning model (Random Forest Classifier).
- **`train_model.py`**: Python script for training the Random Forest model using the dataset.
- **`dataset/`**: Contains the heart disease dataset (`heart.csv`).
- **`templates/`**: HTML templates for the web app:
  - `index.html`: The input form for user data.
  - `result.html`: Displays prediction results.
- **`Project.ipynb`**: Jupyter Notebook for model training and exploratory data analysis.

---

## 📊 Dataset
- **Source**: Heart Disease dataset from Kaggle.
- **Location**: `dataset/heart.csv`
- **Features**:
  - **Age**: Patient's age in years.
  - **Sex**: 0 = Female, 1 = Male.
  - **Chest Pain Type**: Typical angina, atypical angina, non-anginal pain, asymptomatic.
  - **Resting Blood Pressure**: Measured in mmHg.
  - **Cholesterol**: Serum cholesterol level in mg/dl.
  - And more...

---

## ⚙️ Installation
To run the project locally, follow these steps:

### Prerequisites
- Python 3.6 or higher
- pip (Python package installer)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction

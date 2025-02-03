import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load Dataset
data_path = "C:\\Users\\HP\\Documents\\Heart-Disease-Prediction\\dataset\\heart.csv"
df = pd.read_csv(data_path)

# Define Features and Target
X = df.drop(columns=["target"])  # Input features
y = df["target"]  # Target variable

# Normalize the Features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)  # Fit and transform the scaler

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Train the Model
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Test Accuracy
y_pred = model_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the Model and Scaler
model_path = "C:\\Users\\HP\\Documents\\Heart-Disease-Prediction\\heart_disease.pkl"
scaler_path = "C:\\Users\\HP\\Documents\\Heart-Disease-Prediction\\scaler.pkl"

joblib.dump(model_rf, model_path)  # Save the model
joblib.dump(scaler, scaler_path)  # Save the scaler
print(f"Model saved at: {model_path}")
print(f"Scaler saved at: {scaler_path}")
print("DHIVYA A 24BAD020")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve

# Load dataset (YOUR FILE PATH)
data = pd.read_csv(
    r"E:\\ASSIGNMENTS\\ML\\DATASETS\\LICI - 10 minute data.csv"
)

# Clean column names
data.columns = data.columns.str.strip().str.lower()

# Automatically find close column
close_col = [c for c in data.columns if 'close' in c][0]

# Create target variable
data["price_movement"] = np.where(
    data[close_col] > data["open"], 1, 0
)

# Select input features and target
X = data[["open", "high", "low", "volume"]]
y = data["price_movement"]

# Handle missing values
X = X.fillna(X.mean())

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation output
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------- GRAPH 1: ROC CURVE ----------------
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - LIC Stock Price Movement")
plt.show()

# ---------------- GRAPH 2: FEATURE IMPORTANCE ----------------
features = ["Open", "High", "Low", "Volume"]
importance = model.coef_[0]

plt.figure()
plt.barh(features, importance)
plt.xlabel("Coefficient Value")
plt.ylabel("Features")
plt.title("Feature Importance - Logistic Regression")
plt.show()

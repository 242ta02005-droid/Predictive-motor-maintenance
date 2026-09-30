# Predictive-motor-maintenance
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Training data
data = {
    "Temperature": [40, 45, 50, 70, 75, 80, 42, 48, 72, 78],
    "Vibration":   [2, 3, 2, 7, 8, 9, 2, 3, 8, 9],
    "Current":     [5, 5, 6, 8, 9, 10, 5, 6, 9, 10],
    "Hours":       [100, 150, 200, 700, 800, 900, 120, 250, 750, 850],
    "Maintenance": [0, 0, 0, 1, 1, 1, 0, 0, 1, 1]
}

df = pd.DataFrame(data)

# Features
X = df[["Temperature", "Vibration", "Current", "Hours"]]

# Target
y = df["Maintenance"]

# Create and train AI model
model = DecisionTreeClassifier()
model.fit(X, y)

# Get current motor data
temperature = float(input("Enter motor temperature (°C): "))
vibration = float(input("Enter vibration value: "))
current = float(input("Enter motor current (A): "))
hours = float(input("Enter operating hours: "))

# Prediction
prediction = model.predict(
    [[temperature, vibration, current, hours]]
)

# Result
if prediction[0] == 1:
    print("\n⚠ Predictive Maintenance Required!")
    print("Motor may require inspection or servicing.")
else:
    print("\n✓ Motor Condition is Normal.")
    print("Continue regular monitoring.")

import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

# Features: [square_feet, rooms]
X = np.array([
    [500, 1],
    [800, 2],
    [1000, 3],
    [1200, 3],
    [1500, 4],
    [1800, 4],
])

# Prices (example dataset)
y = np.array([
    20000,
    35000,
    50000,
    60000,
    75000,
    90000
])

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model trained successfully and saved as model.pkl")
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

# We train using "form strength difference"

X = np.array([
    [3.5],  # strong home advantage
    [-2.0],
    [1.0],
    [-1.5],
    [2.5],
    [-3.0],
])

y = np.array([1, 0, 1, 0, 1, 0])

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("MatchMind AI trained with form system")
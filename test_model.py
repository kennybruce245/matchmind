import joblib

model = joblib.load("model.pkl")

result = model.predict([[7]])

print(result)
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import r2_score
import pickle

# load data
df = pd.read_csv("Housing.csv")

# features and target
X = df[["area", "bedrooms", "stories", "mainroad",
        "guestroom", "basement",
         "parking"]]


y = df["price"]
#convert text into numbers
binary_cols = ["mainroad","guestroom","basement"]
for col in binary_cols:
    X[col] =  X[col].map({"yes":1,"no":0})



# regression models
models = {
    "Linear": LinearRegression(),
    "RandomForest": RandomForestRegressor(),
    "DecisionTree": DecisionTreeRegressor()
}

best_score = float('-inf')
best_model = None
best_name = ""

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    avg_score = scores.mean()

    print(f"{name}Accuracy {avg_score}")

    if avg_score > best_score:
        best_score = avg_score
        best_model = model
        best_name = name

print("Best model:", best_name)
print("Best Score:", best_score)

# train best model
best_model.fit(X, y)

# save model
pickle.dump(best_model, open("house_model.pkl", "wb"))

print("Best model saved!")







































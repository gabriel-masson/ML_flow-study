# %%
import pandas as pd
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
# %%
model = mlflow.sklearn.load_model("models:/churn-tmw/1")
# %%
print(model.feature_names_in_)

# %%
df = pd.read_csv("./data/abt.csv", sep=",")
df
# %%
X = df.head()[model.feature_names_in_]
X
# %%
proba = model.predict_proba(X)
proba
# %%

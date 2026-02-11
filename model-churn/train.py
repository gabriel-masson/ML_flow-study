# %%
# Execute com shift+enter
import mlflow
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import tree
from sklearn import metrics
print("Imports concluídos.")

# %%
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment(experiment_id=2)


# %%

df = pd.read_csv("./data/abt.csv", sep=",")
features = df.columns[2:-1]
target = "flag_churn"

x = df[features]
y = df[target]

df.head()
#  %%
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.2, random_state=42
)

print(f'Taxa de churn no treinamento: {y_train.mean():.2%}')
print(f'Taxa de churn no teste: {y_test.mean():.2%}')
# %%


# %%
max_depth_array = [3, 30, 60, 120]

for max_depth in max_depth_array:
    with mlflow.start_run():
        # Coom essa linha o mlflow captura todos os parâmetros, métricas e o modelo treinado automaticamente e os registra
        mlflow.sklearn.autolog()
        clf = tree.DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=20,
            random_state=42)
        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        acc_train = metrics.accuracy_score(y_train, y_train_pred)
        acc_test = metrics.accuracy_score(y_test, y_test_pred)

        mlflow.log_metrics({
            "acc_train": acc_train,
            "acc_test": acc_test
        })


# %%
print(f"Acurácia no treinamento: {acc_train:.2%}")
print(f"Acurácia no teste: {acc_test:.2%}")
# %%

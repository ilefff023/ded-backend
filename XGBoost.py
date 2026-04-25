import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Bibliothèques de Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import learning_curve


FILE_PATH = r"C:\Users\PC\Desktop\data raw\final\DED_Final_Sensor_Fusion.csv"
df = pd.read_csv(FILE_PATH)

# Séparation des caractéristiques (X) et de la cible (y)
X = df.drop('Condition', axis=1)
y = df['Condition']

# Découpage Stratifié (80% entraînement, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalisation (Nécessaire uniquement pour le SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("🚀 Démarrage de la comparaison des modèles...\n")
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb) * 100
print(f"🎯 Accuracy XGBoost : {accuracy_xgb:.2f}%")
print(classification_report(y_test, y_pred_xgb))


# 1. Déplacez eval_metric dans l'initialisation
xgb_model = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    random_state=42,
    eval_metric="merror"  # <--- On le met ici maintenant
)

# 2. L'entraînement (fit) ne garde que eval_set et verbose
eval_set = [(X_train, y_train), (X_test, y_test)]
xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)


# Récupération des logs
results = xgb_model.evals_result()
epochs = len(results['validation_0']['merror'])
x_axis = range(0, epochs)

plt.figure(figsize=(8, 5))
plt.plot(x_axis, results['validation_0']['merror'], label='Train Error')
plt.plot(x_axis, results['validation_1']['merror'], label='Test/Val Error')
plt.title('XGBoost : Erreur de classification vs Itérations')
plt.xlabel('Nombre d\'arbres (N-estimators)')
plt.ylabel('Erreur (Merror)')
plt.legend()
plt.grid(True)
plt.show()
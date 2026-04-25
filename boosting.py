import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# --- 1. CHARGEMENT DES DONNÉES ---
FILE_PATH = r"C:\Users\PC\Desktop\data raw\final\DED_Final_Sensor_Fusion.csv"
df = pd.read_csv(FILE_PATH)

X = df.drop('Condition', axis=1)
y = df['Condition']

# --- 2. ENCODAGE DES LABELS ---
le = LabelEncoder()
y = le.fit_transform(y)
# CORRECTION : On s'assure que les noms sont des strings pour éviter l'erreur TypeError
class_names = [str(c) for c in le.classes_]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3. CONFIGURATION ET ENTRAÎNEMENT ---
print(f"🚀 Analyse des données avec XGBoost (Classes : {class_names})...\n")

xgb_model = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    random_state=42,
    eval_metric="merror"
    # use_label_encoder=False  <-- Supprimé car inutile dans les versions récentes
)

eval_set = [(X_train, y_train), (X_test, y_test)]

xgb_model.fit(
    X_train, y_train, 
    eval_set=eval_set, 
    verbose=False 
)

# --- 4. PRÉDICTIONS ET ÉVALUATION ---
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

print(f"🎯 Accuracy Finale XGBoost : {accuracy:.2f}%")
print("\n--- RAPPORT DE CLASSIFICATION ---")
# Utilise maintenant la liste class_names convertie en strings
print(classification_report(y_test, y_pred, target_names=class_names))

# --- 5. VISUALISATION DES RÉSULTATS ---
results = xgb_model.evals_result()
epochs = len(results['validation_0']['merror'])
x_axis = range(0, epochs)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_axis, results['validation_0']['merror'], label='Train Error')
plt.plot(x_axis, results['validation_1']['merror'], label='Test Error')
plt.title('Erreur de Classification')
plt.xlabel('Nombre d\'itérations')
plt.ylabel('Erreur')
plt.legend()

plt.subplot(1, 2, 2)
cm = confusion_matrix(y_test, y_pred)
# On utilise class_names pour l'affichage visuel également
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), colorbar=False)
plt.title('Matrice de Confusion')

plt.tight_layout()
plt.show()

# --- 6. IMPORTANCE DES CARACTÉRISTIQUES ---
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh', color='teal')
plt.title('Importance des capteurs dans la décision finale')
plt.gca().invert_yaxis() # Pour avoir la plus importante en haut
plt.show()

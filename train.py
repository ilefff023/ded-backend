import pandas as pd
import joblib
from xgboost import XGBClassifier

# 1. Tes données d'entraînement
#    Colonnes attendues : temp, humidity, lux, eye_temp, blink_rate
#    Label : 0=normal, 1=moderate, 2=severe
FILE_PATH = r"C:\Users\PC\Desktop\data raw\final\DED_Final_Sensor_Fusion.csv"
df = pd.read_csv(FILE_PATH)

X = df[["Amb_Temp", "Amb_Humidity", "Luminosity", "UnderEye_Temp", "Blink_Rate"]]
y = df["Condition"]   # 0, 1, ou 2

# 2. Entraînement
model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
model.fit(X, y)

# 3. Sauvegarde → génère model.pkl
joblib.dump(model, "model.pkl")
print("model.pkl créé")
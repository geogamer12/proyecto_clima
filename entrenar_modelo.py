import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

CSV_HISTORICO = "barranquilla_historico.csv"

print("Cargando 'barranquilla_historico.csv'...")
df = pd.read_csv(CSV_HISTORICO)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour

df["target"] = df["temp"].shift(-1)
df = df.dropna()

features = ["temp", "humedad", "presion", "hour"]
X = df[features]
y = df["target"]

split = int(len(df) * 0.8)
X_train, y_train = X.iloc[:split], y.iloc[:split]
X_test, y_test = X.iloc[split:], y.iloc[split:]

print(f"Entrenando XGBoost con {len(X_train)} registros historicos...")

modelo = XGBRegressor(
    n_estimators=150,
    learning_rate=0.03,
    max_depth=5,
    random_state=42
)
modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)
mae = mean_absolute_error(y_test, predicciones)
r2 = r2_score(y_test, predicciones)

print("\n==================================================")
print("RESULTADOS DEL ENTRENAMIENTO")
print("==================================================")
print(f"Error Absoluto Medio (MAE): {mae:.3f} °C")
print(f"Coeficiente de Determinacion (R2): {r2:.4f}")
print("==================================================")

JOBLIB_FILE = "modelo_clima_xgboost.joblib"
joblib.dump(modelo, JOBLIB_FILE)
print(f"Modelo guardado con exito como '{JOBLIB_FILE}'.")
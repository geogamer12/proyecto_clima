import os
import requests
import joblib
import pandas as pd
from datetime import datetime

LATITUD = 10.9685
LONGITUD = -74.7813
MODELO_PATH = "modelo_clima_xgboost.joblib"
CSV_PATH = "registro_clima.csv"

def capturar_clima_actual():
    url = f"https://api.open-meteo.com/v1/forecast?latitude={LATITUD}&longitude={LONGITUD}&current=temperature_2m,relative_humidity_2m,surface_pressure&timezone=America/Bogota"
    res = requests.get(url, timeout=10)
    res.raise_for_status()
    current = res.json()["current"]
    
    now = datetime.now()
    return {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "hour": now.hour,
        "temp_real": current["temperature_2m"],
        "humedad_real": current["relative_humidity_2m"],
        "presion_real": current["surface_pressure"]
    }

def main():
    if not os.path.exists(MODELO_PATH):
        print("Error: No se encuentra 'modelo_clima_xgboost.joblib'. Ejecuta primero entrenar_modelo.py")
        return

    modelo = joblib.load(MODELO_PATH)

    clima = capturar_clima_actual()

    input_data = pd.DataFrame([[
        clima["temp_real"], 
        clima["humedad_real"], 
        clima["presion_real"], 
        clima["hour"]
    ]], columns=["temp", "humedad", "presion", "hour"])

    prediccion = float(modelo.predict(input_data)[0])
    clima["prediccion_ia"] = round(prediccion, 2)

    df_fila = pd.DataFrame([{
        "timestamp": clima["timestamp"],
        "temp_real": clima["temp_real"],
        "humedad_real": clima["humedad_real"],
        "presion_real": clima["presion_real"],
        "prediccion_ia": clima["prediccion_ia"]
    }])

    hdr = not os.path.exists(CSV_PATH)
    df_fila.to_csv(CSV_PATH, mode="a", header=hdr, index=False)
    
    print(f"[{clima['timestamp']}] Temp Real: {clima['temp_real']}°C | Prediccion XGBoost (t+1h): {clima['prediccion_ia']}°C")

if __name__ == "__main__":
    main()
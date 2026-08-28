import requests
import pandas as pd

LATITUD = 10.9685
LONGITUD = -74.7813

URL = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": LATITUD,
    "longitude": LONGITUD,
    "start_date": "2026-08-01",
    "end_date": "2026-08-28",
    "hourly": ["temperature_2m", "relative_humidity_2m", "surface_pressure"],
    "timezone": "America/Bogota"
}

res = requests.get(URL, params=params, timeout=20)
res.raise_for_status()
data = res.json()["hourly"]

df = pd.DataFrame({
    "timestamp": data["time"],
    "temp": data["temperature_2m"],
    "humedad": data["relative_humidity_2m"],
    "presion": data["surface_pressure"]
})

df.to_csv("barranquilla_historico.csv", index=False)
print(f"Se descargaron {len(df)} registros correctamente en 'barranquilla_historico.csv'.")
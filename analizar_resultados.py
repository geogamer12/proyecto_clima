import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CSV_PATH = "registro_clima.csv"

if not os.path.exists(CSV_PATH):
    print("No existe aun el archivo 'registro_clima.csv'. Ejecuta logger_clima_once.py algunas veces.")
else:
    df = pd.read_csv(CSV_PATH)

    if len(df) < 4:
        print("Se necesitan mas registros acumulados para un analisis estadistico.")
    else:
        df["temp_real_siguiente"] = df["temp_real"].shift(-4)
        eval_df = df.dropna()

        y_real = eval_df["temp_real_siguiente"]
        y_pred = eval_df["prediccion_ia"]

        mae = mean_absolute_error(y_real, y_pred)
        rmse = np.sqrt(mean_squared_error(y_real, y_pred))
        r2 = r2_score(y_real, y_pred)
        mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
        precision = 100 - mape

        print("==================================================")
        print("AUDITORIA CUANTITATIVA DEL RENDIMIENTO")
        print("==================================================")
        print(f"Muestras Evaluadas: {len(eval_df)}")
        print(f"MAE: {mae:.3f} °C")
        print(f"RMSE: {rmse:.3f} °C")
        print(f"R2: {r2:.4f}")
        print(f"MAPE: {mape:.2f} %")
        print(f"Precision Global: {precision:.2f} %")
        print("==================================================")

        plt.figure(figsize=(10, 5))
        plt.plot(eval_df["timestamp"], y_real, label="Real (t+1h)", color="#1f77b4", marker="o")
        plt.plot(eval_df["timestamp"], y_pred, label="Prediccion XGBoost", color="#ff7f0e", linestyle="--", marker="s")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Evaluacion del Modelo - Precision: {precision:.1f}% (MAE: {mae:.2f}°C)")
        plt.xlabel("Fecha / Hora")
        plt.ylabel("Temperatura (°C)")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig("grafica_rendimiento.png")
        print("Grafica guardada como 'grafica_rendimiento.png'.")
        plt.show()
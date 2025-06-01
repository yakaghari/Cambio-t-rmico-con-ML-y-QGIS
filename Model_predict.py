# Script: Model_predict.py
# Autor: Yassine Kachoua Ghari
# Fecha: 06/06/2025
# Proyecto: TFM - Análisis del cambio térmico mediante Machine Learning y visualizaciones QGIS:
# Comparativa entre países desarrollados y en desarrollo.

# Nota para ejecutar los codigos hay que cambiar la ubicación de carga de los archivos de datos según el caso. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


# Configuración visual
sns.set(style="whitegrid")

# Cargar el dataset
df = pd.read_csv("df_final.csv")

# Verificar las primeras filas
df.head()

# Boxplot de anomalías por país (última década)
df_last_decade = df[df["year"] >= 2010]
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_last_decade, x="country", y="Temp_Anomaly", palette="viridis")
plt.title("Distribución de Anomalías Térmicas (2010-2020)")
plt.xticks(rotation=45)
plt.show()

# Regresión lineal 

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df[["GDP", "co2", "year"]]
y = df["Temp_Anomaly"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))


# Random Forest 

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))
print("Importancia de variables:", rf.feature_importances_)

# Randon Forest sin year

X = df[["GDP", "co2"]]
y = df["Temp_Anomaly"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)

print("R²:", r2_score(y_test, y_pred))
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f} °C")
print("Importancia de variables:", rf.feature_importances_)


# Red neuronal

from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)
y_pred = mlp.predict(X_test_scaled)

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))

# Análisis por grupo o nivel de desarrollo

# Clasificación sigún nivel de desarrollo
grupo_desarrollado = ["France", "Germany", "Japan"]
grupo_en_desarrollo = ["Brazil", "Indonesia", "Nigeria"]

df["grupo"] = df["country"].apply(
    lambda x: "Desarrollado" if x in grupo_desarrollado else "En desarrollo"
)
df_desarrollado = df[df["grupo"] == "Desarrollado"]
df_en_desarrollo = df[df["grupo"] == "En desarrollo"]

def entrenar_modelo_rf(dataframe, grupo_nombre):
    X = dataframe[["GDP", "co2", "year"]]
    y = dataframe["Temp_Anomaly"]
    
    # División
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelo
    modelo = RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_split=5, random_state=42)
    modelo.fit(X_train_scaled, y_train)

    # Evaluación
    y_pred = modelo.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    importancia = modelo.feature_importances_

    print(f"\nResultados para: {grupo_nombre}")
    print(f"  RMSE: {rmse:.4f} °C")
    print(f"  R²: {r2:.4f}")
    print("  Importancia de variables:")
    for var, imp in zip(["GDP", "co2", "year"], importancia):
        print(f"   - {var}: {imp:.3f}")

entrenar_modelo_rf(df_desarrollado, "Desarrollado")
entrenar_modelo_rf(df_en_desarrollo, "En desarrollo")

# Ilustrar la relación entre el desarrollo económico y las emisiones
df["group"] = df["country"].apply(lambda x: "developed" if x in ["France", "Japan", "Germany"] else "developing")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="GDP", y="co2", hue="group", style="country", s=100)
plt.title("Relación GDP vs Emisiones CO₂ por País")
plt.xlabel("GDP (USD)")
plt.ylabel("CO₂ (toneladas)")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.show()


# Predicción temporal del modelo

# Regresión lineal 

# Filtrar solo los años hasta 2023 (por si hay valores futuros añadidos más adelante)
df = df[df["year"] <= 2023]

# Separar datos de entrenamiento (hasta 2020) y de predicción (2021–2023)
df_train = df[df["year"] <= 2020]
df_test_future = df[df["year"] > 2020]

# Variables predictoras y objetivo
features = ["GDP", "co2", "year"]
target = "Temp_Anomaly"

X_train = df_train[features]
y_train = df_train[target]

X_future = df_test_future[features]
y_future_real = df_test_future[target] 

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_future_scaled = scaler.transform(X_future)

# Entrenar modelo
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Predecir los años 2021, 2022 y 2023
y_future_pred = lr.predict(X_future_scaled)

# Evaluación del rendimiento sobre los años recientes
rmse = np.sqrt(mean_squared_error(y_future_real, y_future_pred))
r2 = r2_score(y_future_real, y_future_pred)

print(f"RMSE (2021–2023): {rmse:.4f} °C")
print(f"R² (2021–2023): {r2:.4f}")

# Gráfica: Real / Predicción
years = df_test_future["year"]
plt.figure(figsize=(10, 5))
plt.plot(years, y_future_real, label="Real", marker="o", linewidth=2)
plt.plot(years, y_future_pred, label="Predicción", marker="s", linestyle="--", linewidth=2)
plt.xlabel("Año")
plt.ylabel("Anomalía de Temperatura (°C)")
plt.title("Comparación: Real vs Predicción (2021–2023)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Random Forest 

# Dividir en entrenamiento y prueba
df_train = df[df["year"] <= 2020]
df_test = df[df["year"] > 2020]

X_train = df_train[["GDP", "co2", "year"]]
y_train = df_train["Temp_Anomaly"]

X_test = df_test[["GDP", "co2", "year"]]
y_test = df_test["Temp_Anomaly"]

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Predecir
y_pred = rf.predict(X_test_scaled)

# Evaluar
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE (2021–2023):", round(rmse, 4), "°C")
print("R² (2021–2023):", round(r2, 4))

# Visualizar 
plt.figure(figsize=(8,5))
plt.plot(y_test.values, label="Real", marker='o')
plt.plot(y_pred, label="Predicción RF", marker='x')
plt.title("Comparación Real vs Predicción (2021–2023) – Random Forest")
plt.xlabel("Observación")
plt.ylabel("Anomalía de Temperatura")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Red neuronal

# Modelo MLP
mlp = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Predicir
y_pred = mlp.predict(X_test_scaled)

# Evaluar
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE (2021–2023):", round(rmse, 4), "°C")
print("R² (2021–2023):", round(r2, 4))

# Visualizar
plt.figure(figsize=(8, 5))
plt.plot(y_test.values, label="Real", marker='o')
plt.plot(y_pred, label="Predicción MLP", marker='x')
plt.title("Comparación Real vs Predicción (MLP) - 2021 a 2023")
plt.xlabel("Índice de observación")
plt.ylabel("Anomalía de temperatura")
plt.legend()
plt.grid(True)
plt.show()



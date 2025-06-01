# Script: Analis_explo.R
# Descripción: Análisis exploratorio de los datos
# Autor: Yassine Kachoua Ghari
# Fecha: 06/06/2025
# Proyecto: TFM - Análisis del cambio térmico mediante Machine Learning y visualizaciones QGIS:
# Comparativa entre países desarrollados y en desarrollo.

# Nota para ejecutar los codigos hay que cambiar la ubicación de carga de los archivos de datos según el caso. 

library(readr)

# Cargar el dataset final
path <- "C:/Users/user/Desktop/Master Ciencias de Datos/TFM/fase 3/Desarrollo/Datos procesados/Datos_Limpios/df_final.csv"
df <- read_csv(file = path)

# Mostrar primeros estadísticos
summary(df)

# Visualizar la tendencia anual de las tres variables 
library(ggplot2)
library(dplyr)

# Agregamos los datos por año (promedio para todos los países)
df_yearly <- df %>%
  group_by(year) %>%
  summarise(
    mean_co2 = mean(co2, na.rm = TRUE),
    mean_gdp = mean(GDP, na.rm = TRUE),
    mean_temp = mean(Temp_Anomaly, na.rm = TRUE)
  )

# Tendencia anual de emisiones de CO2
ggplot(df_yearly, aes(x = year, y = mean_co2)) +
  geom_line(color = "darkred", size = 1) +
  labs(title = "Tendencia Anual de Emisiones de CO₂ (Media Global)",
       x = "Año", y = "Emisiones CO₂ (Toneladas per cápita)") +
  theme_minimal()

# Tendencia anual del PIB
ggplot(df_yearly, aes(x = year, y = mean_gdp)) +
  geom_line(color = "steelblue", size = 1) +
  labs(title = "Tendencia Anual del PIB per cápita (Media Global)",
       x = "Año", y = "PIB (USD)") +
  theme_minimal()

# Tendencia anual de anomalía de temperatura
ggplot(df_yearly, aes(x = year, y = mean_temp)) +
  geom_line(color = "darkgreen", size = 1) +
  labs(title = "Tendencia Anual de Anomalía de Temperatura (Media Global)",
       x = "Año", y = "Anomalía (°C)") +
  theme_minimal()


# Visualizar la evolución de emisiones por grupo de países

# Clasificar países en categorías
df$grupo <- ifelse(df$country %in% c("France", "Germany", "Japan"), "Desarrollado", "En desarrollo")

# Gráfico de líneas: Emisiones CO₂ por grupo
df %>%
  group_by(grupo, year) %>%
  summarise(media_co2 = mean(co2, na.rm = TRUE)) %>%
  ggplot(aes(x = year, y = media_co2, color = grupo)) +
  geom_line(size = 1.2) +
  labs(title = "Evolución de Emisiones de CO₂ por Grupo de Países",
       x = "Año", y = "Media de Emisiones CO₂ (toneladas per cápita)") +
  theme_minimal()

# Visualizar la evolución del PIB por grupo de países

df %>%
  group_by(grupo, year) %>%
  summarise(media_gdp = mean(GDP, na.rm = TRUE)) %>%
  ggplot(aes(x = year, y = media_gdp, color = grupo)) +
  geom_line(size = 1.2) +
  labs(title = "Evolución del PIB per cápita por Grupo de Países",
       x = "Año", y = "PIB per cápita (USD)") +
  theme_minimal()

# Visualizar la evolución de la anomalía de temperatura por grupo de países

df %>%
  group_by(grupo, year) %>%
  summarise(media_temp = mean(Temp_Anomaly, na.rm = TRUE)) %>%
  ggplot(aes(x = year, y = media_temp, color = grupo)) +
  geom_line(size = 1.2) +
  labs(title = "Evolución de la Anomalía de Temperatura por Grupo de Países",
       x = "Año", y = "Anomalía de temperatura (°C)") +
  theme_minimal()




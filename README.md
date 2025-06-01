# Análisis del cambio térmico mediante Machine Learning y visualizaciones QGIS: Comparativa entre países desarrollados y en desarrollo.

## Autor: Yassine Kachoua Ghari  
## Máster en Ciencia de Datos - Universidad Oberta de Catalunya 
## Junio 2025

## Descripción del proyecto

Este Trabajo Final de Máster (TFM) tiene como objetivo analizar y predecir la evolución de la temperatura media global utilizando técnicas de Machine Learning, visualización de datos y sistemas de información geográfica (GIS) (Incluido exclusivamente en la memoria del TFM). El estudio compara seis países (Alemania, Francia, Japón, Brasil, Indonesia y Nigeria) en función de sus niveles de desarrollo económico, utilizando datos de emisiones de CO₂, PIB per cápita y anomalías térmicas anuales.

## Tecnologías utilizadas
- R + RStudio: Limpieza, transformación y exploración inicial de datos.
- Python (scikit-learn, pandas, numpy, matplotlib): Modelado predictivo.
- QGIS: Visualización geoespacial de mapas temáticos.

## Modelos aplicados
Se aplicaron y compararon los siguientes algoritmos para predecir la anomalía de temperatura:

- Regresión Lineal Múltiple

- Random Forest Regressor

- Red Neuronal MLP (Multilayer Perceptron)

También se aplicó un análisis por subgrupos (desarrollados vs en desarrollo) y una predicción temporal específica para los años 2021–2023.

Resultados principales
Modelo	RMSE (°C)	R²
Regresión Lineal	0.6993	0.2247
Random Forest	0.7505	0.1071
Red Neuronal (MLP)	0.7632	0.0766

Random Forest mostró el mejor rendimiento fuera de muestra (2021–2023).

## Fuentes de datos

- CO₂: Our World in Data

- PIB: World Bank (Banco Mundial)

- Temperatura: Berkeley Earth High-Resolution Dataset

## Licencia
Este proyecto es de uso académico. Si desea reutilizar parte del código o los datos, por favor cite este repositorio o contacte con el autor.

# Script: data_prep.R
# Descripción: Limpieza y transformación de datos para el análisis climático
# Autor: Yassine Kachoua Ghari
# Fecha: 06/06/2025
# Proyecto: TFM - Análisis del cambio térmico mediante Machine Learning y visualizaciones QGIS:
# Comparativa entre países desarrollados y en desarrollo.


# Nota para ejecutar los códigos hay que cambiar la ubicación de carga de los archivos de datos según el caso. 

############ DATOS PIB ################
# Aumentar tamaño del buffer de lectura
Sys.setenv("VROOM_CONNECTION_SIZE" = 5000000)

library(readr)

path2 <- "C:/Users/user/Desktop/Master Ciencias de Datos/TFM/fase 3/Desarrollo/Datos crudos/PBI_GDP/API_c.csv"

# Leer el archivo de datos
df <- read_csv(
  file = path2,
  col_types = cols(.default = "c"),  # Leer como texto
  quote = "\""                       # Respetar comillas dobles
)

colnames(df)[1:10] 
# Leer el archivo como texto
lines <- readLines("C:/Users/user/Desktop/Master Ciencias de Datos/TFM/fase 3/Desarrollo/Datos crudos/PBI_GDP/API_c.csv", encoding = "UTF-8")

# Eliminar las comillas dobles
API <- gsub('"', '', lines)

# Guardar el resultado en un nuevo archivo
writeLines(API, "C:/Users/user/Desktop/Master Ciencias de Datos/TFM/fase 3/Desarrollo/Datos crudos/PBI_GDP/API.csv")
# Cargar los datos del nuevo arvhico CSV
df <- read.csv("C:/Users/user/Desktop/Master Ciencias de Datos/TFM/fase 3/Desarrollo/Datos crudos/PBI_GDP/API.csv", stringsAsFactors = FALSE)
head(df)

# Quitar la "X" de los nombres de columna que empiecen con "X" y estén seguidas por 4 dígitos
 names(df) <- gsub("^X(\\d{4})$", "\\1", names(df))

# Eliminar columnas por nombre
df <- df[, !names(df) %in% c("X","1960")]

# Reemplazar NA y valores vacíos por 0
df[is.na(df) | df == ""] <- 0
# Convertir columnas año a tipo numérico
df_clean <- df %>%
  mutate_at(vars(matches("^[0-9]{4}$")), as.numeric)

# Convertir los valores a formato largo
library(tidyr)
df_largo <- df %>%
  pivot_longer(
    cols = matches("^[0-9]{4}$"),
    names_to = "Year",
    values_to = "GDP"
  )

head(df_largo, 3)

# Eliminar columnas por nombre
df_largo <- df_largo[, !names(df_largo) %in% c("Country.Code","Indicator.Name","Indicator.Code")]
head(df_largo, 3)

# Guardar
write.csv(df_largo, "C:/Users/user/Desktop/Master Ciencias de Datos/TFM/fase 3/Desarrollo/Datos crudos/PBI_GDP/PIB_Clean.csv", row.names = FALSE)

############ DATOS Temp_Anomaly ################
library(dplyr)
library(readr)

procesar_temperatura <- function(path_txt, country_name) {
  # Leer el archivo 
  data <- read_table(
    file = path_txt,
    skip = 2, 
    col_names = c("Year", "Month", "MonthlyAnomaly", "MonthlyUnc",
                  "AnnualAnomaly", "AnnualUnc", 
                  "FiveYearAnomaly", "FiveYearUnc",
                  "TenYearAnomaly", "TenYearUnc",
                  "TwentyYearAnomaly", "TwentyYearUnc"),
    na = "NaN"   # Leer 'NaN' como faltantes
  )
  
  # Seleccionar las columnas que nos interesan
  data_clean <- data %>%
  select(Year, Month, MonthlyAnomaly) %>%
  mutate(
    Year = as.numeric(Year),
    Country = country_name)
  
  # Calcular promedio anual
  annual_data <- data_clean %>%
    group_by(Country, Year) %>%
    summarise(Temp_Anomaly = mean(MonthlyAnomaly, na.rm = TRUE), .groups = 'drop') %>%
    filter(Year >= 1950)  # Limitar a partir de 1950
  
  return(annual_data)
}

# Paths de archivos
path_germany <- "germany-TAVG-monthly.txt"
path_brazil <- "brazil-TAVG-monthly.txt"
path_nigeria <- "nigeria-TAVG-monthly.txt"
path_india <- "indonesia-TAVG-monthly.txt"
path_japan <- "japan-TAVG-monthly.txt"
path_france <- "france-TAVG-monthly.txt"

# Aplicar
germany_temp <- procesar_temperatura(path_germany, "Germany")
brazil_temp <- procesar_temperatura(path_brazil, "Brazil")
nigeria_temp <- procesar_temperatura(path_nigeria, "Nigeria")
india_temp <- procesar_temperatura(path_india, "Indonesia")
japan_temp <- procesar_temperatura(path_japan, "Japan")
france_temp <- procesar_temperatura(path_france, "France")

# Unir
temperatura_final <- bind_rows(germany_temp, brazil_temp, nigeria_temp, india_temp, japan_temp, france_temp)

# Mostrar
head(temperatura_final)

# Guardar
write_csv(temperatura_final, "Temperatura_Paises_Anual.csv")

############ DATOS CO2 ################
library(readr)
# Lectura y Carga de datos
path <- "C:/Users/user/Desktop/Master Ciencias de Datos/TFM/fase 3/Desarrollo/Datos crudos/CO2_Emisiones/co2_data.csv"
df <- read_csv(file = path)
# Seleccionar las columnas deseadas
df <- df[, c("country", "year", "co2")]

# Guardar el archivo de datos de co2
write.csv(df, "C:/Users/user/Desktop/Master Ciencias de Datos/TFM/fase 3/Desarrollo/Datos crudos/CO2_Emisiones/co2.csv", row.names = FALSE) 

############## UNIR LOS CONJUNTOS DE DATOS Y GUARDAR EN UN DATASET FINAL ################### 
# Cargar los datos
co2 <- read_csv("C:/Users/user/Desktop/Master Ciencias de Datos/TFM/fase 3/Desarrollo/Datos crudos/CO2_Emisiones/co2.csv")
pib <- read_csv("C:/Users/user/Desktop/Master Ciencias de Datos/TFM/fase 3/Desarrollo/Datos crudos/PBI_GDP/PIB_Clean.csv")
temp <- read_csv("C:/Users/user/Desktop/Master Ciencias de Datos/TFM/fase 3/Desarrollo/Datos crudos/MODIS_Temperatura/Countries/Temperatura_Paises_Anual.csv")

# Estándarizar nombres de columnas
pib <- pib %>%
  rename(country = Country.Name,
         year = Year) %>%
  mutate(year = as.numeric(year))

temp <- temp %>%
  rename(country = Country,
         year = Year)

# Unir los datasets
df_final <- co2 %>%
  inner_join(pib, by = c("country", "year")) %>%
  inner_join(temp, by = c("country", "year"))

# Verificar
glimpse(df_final)

# Guardar
write.csv(df_final, "C:/Users/user/Desktop/Master Ciencias de Datos/TFM/fase 3/Desarrollo/Datos procesados/Datos_Limpios/df_final.csv", row.names = FALSE)





# Translog_XML_Analyzer_py
# Author                    : Dr. Marcos H. Cárdenas Mancilla
# E-mail                    : marcos.cardenas.m@usach.cl
# Date of creation          : 2024-11-16
# Licence                   : AGPL V3
# Copyright (c) 2024 Marcos H. Cárdenas Mancilla.

# Descripción de Translog_XML_Analyzer_py:
# Este script de Python analiza datos de tiempo de respuesta (RT) extraídos de archivos XML que fueron generados Translog II (versión 2.0) (Carl, 2012). 
# El código procesa múltiples archivos XML para el análisis comparativo de RTs entre diferentes grupos de variables intrasujeto e intratarea y sus interacciones.
# Características del pipeline metodologógico:
# 1. extracción información sobre participantes, niveles de experiencia, texto de la tarea de traducción, tipo de evento y acciones realizadas.
# 2. cálculo de los tiempos de respuesta (RT) como la diferencia entre eventos consecutivos.
![RTeventos](https://github.com/user-attachments/assets/02e5f99e-dfcb-4283-85ee-85a305fbf942)
# 3. análisis estadísticos automatizados i.e., pruebas de normalidad (Shapiro-Wilk y D’Agostino-Pearson), Kruskal-Wallis para comparar grupos, y Dunn's post-hoc para comparaciones por pares.
![smaple](https://github.com/user-attachments/assets/4dc4ef7f-34ac-4132-95be-2ce88e5fbba1)
# 4. visualización del resultados del análisis descriptivo (p. ej. tablas, barras, boxplots, mapas de calor).
![mapacalor](https://github.com/user-attachments/assets/10d017b5-4a12-4748-9216-ddbbce63d066)
# 5. cálculo de tamaños de efecto a partir del análisis inferencial (p. ej. correlación biserial de rango) para determinar los efectos significativos que permitan comprender las relaciones entre RT y las variables independientes (p. ej. experiencia en traducción, tipo de acción y texto).
![efectos](https://github.com/user-attachments/assets/7cf8d789-991b-4576-8447-f798a1d88280)

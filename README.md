# EDA-Open-Data-Analysis

Link de archivo colab: https://colab.research.google.com/drive/1EfDai1OVLZor4WL1C40CJHqF8UqdFH72?usp=sharing

# Análisis Exploratorio de Datos (EDA) - Adicción a Redes Sociales en Estudiantes

## 1. Introducción

Este proyecto tiene como objetivo realizar un Análisis Exploratorio de Datos (EDA) sobre un dataset que examina la adicción a las redes sociales en estudiantes. El análisis busca comprender la estructura del dataset, evaluar su calidad y extraer hallazgos relevantes que puedan aportar valor desde una perspectiva de negocio y académica.

## 2. Alcance del EDA

El EDA realizado incluye:
- Análisis cuantitativo de variables numéricas.
- Análisis cualitativo de variables categóricas.
- Análisis gráfico para visualización de distribuciones y relaciones.
- Interpretaciones y conclusiones con enfoque de negocio.

## 3. Configuración del Entorno y Carga de Datos

Para ejecutar este análisis, se requieren las siguientes librerías de Python:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)
sns.set(style="whitegrid")
```

El dataset utilizado se carga desde el archivo `Students Social Media Addiction.csv`:

```python
df = pd.read_csv("Students Social Media Addiction.csv", nrows=10000)
```

## 4. Vista General del Dataset

El dataset cuenta con `705` filas y `13` columnas, ocupando aproximadamente `71.7 KB` de memoria. Contiene una mezcla de tipos de datos (`int64`, `float64`, `object`).

```
Filas, Columnas: (705, 13)
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 705 entries, 0 to 704
Data columns (total 13 columns):
 #   Column                        Non-Null Count  Dtype  
---  ------                        --------------  -----  
 0   Student_ID                    705 non-null    int64  
 1   Age                           705 non-null    int64  
 2   Gender                        705 non-null    object 
 3   Academic_Level                705 non-null    object 
 4   Country                       705 non-null    object 
 5   Avg_Daily_Usage_Hours         705 non-null    float64
 6   Most_Used_Platform            705 non-null    object 
 7   Affects_Academic_Performance  705 non-null    object 
 8   Sleep_Hours_Per_Night         705 non-null    float64
 9   Mental_Health_Score           705 non-null    int64  
 10  Relationship_Status           705 non-null    object 
 11  Conflicts_Over_Social_Media   705 non-null    int64  
 12  Addicted_Score                705 non-null    int64  
dtypes: float64(2), int64(5), object(6)
memory usage: 71.7+ KB
```

### Descripción de Columnas

| Variable                         | Tipo        | Descripción                                                                 |
|----------------------------------|-------------|------------------------------------------------------------------------------|
| Student_ID                       | Entero      | Identificador único del encuestado                                           |
| Age                              | Entero      | Edad en años                                                                 |
| Gender                           | Categórico  | Masculino o Femenino                                                         |
| Academic_Level                   | Categórico  | Nivel académico: Secundaria / Pregrado / Posgrado                            |
| Country                          | Categórico  | País de residencia                                                           |
| Avg_Daily_Usage_Hours            | Decimal     | Promedio de horas diarias en redes sociales                                  |
| Most_Used_Platform               | Categórico  | Plataforma principal utilizada (Instagram, Facebook, TikTok, etc.)          |
| Affects_Academic_Performance     | Booleano    | Impacto auto-reportado de redes sociales en el rendimiento académico (Sí/No)|
| Sleep_Hours_Per_Night            | Decimal     | Promedio de horas de sueño por noche                                         |
| Mental_Health_Score              | Entero      | Puntaje de salud mental (1 = Muy mala, 10 = Excelente)                       |
| Relationship_Status              | Categórico  | Estado de relación: Soltero / En relación / Complicado                       |
| Conflicts_Over_Social_Media      | Entero      | Número de conflictos de pareja causados por redes sociales                  |
| Addicted_Score                   | Entero      | Puntaje de adicción a redes sociales (1 = Bajo, 10 = Alto)                  |

## 5. Hallazgos Generales Preliminares

- La mitad de la muestra se encuentra en el nivel académico de pregrado (Undergraduate).
- Más de la mitad de los encuestados son solteros.
- India es el país de residencia más frecuente entre los encuestados.
- Instagram es la plataforma de redes sociales más utilizada.
- Más de la mitad de los encuestados perciben que el uso de redes sociales afecta su rendimiento académico.
- En promedio, los encuestados reportan un puntaje de adicción a redes sociales de 6.4 (en una escala de 1 a 10).

## 6. Calidad de los Datos

### Valores Nulos

El dataset no presenta valores nulos, lo que indica una buena calidad en la recolección de datos.

```python
missing = df.isnull().mean().sort_values(ascending=False) * 100
missing = missing[missing > 0]
print(missing)

# Visualización de valores nulos (confirmando ausencia)
plt.figure(figsize=(10,4))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Mapa de valores nulos")
plt.show()
```

### Duplicados

No se encontraron filas duplicadas en el dataset.

```python
duplicados = df.duplicated().sum()
print("Duplicados:", duplicados)
```

### Tipos de Datos Inconsistentes

Los tipos de datos de las columnas son consistentes con sus descripciones, sin problemas de inconsistencias.

```python
df.dtypes.value_counts()
```

## 7. Análisis Univariado

### Variables Numéricas

Se calcularon estadísticas descriptivas para las columnas numéricas:

```
                               count        mean         std   min    25%  \
Student_ID                   705.0  353.000000  203.660256   1.0  177.0   
Age                          705.0   20.659574    1.399217  18.0   19.0   
Avg_Daily_Usage_Hours        705.0    4.918723    1.257395   1.5    4.1   
Sleep_Hours_Per_Night        705.0    6.868936    1.126848   3.8    6.0   
Mental_Health_Score          705.0    6.226950    1.105055   4.0    5.0   
Conflicts_Over_Social_Media  705.0    2.849645    0.957968   0.0    2.0   
Addicted_Score               705.0    6.436879    1.587165   2.0    5.0   

                               50%    75%    max  
Student_ID                   353.0  529.0  705.0  
Age                           21.0   22.0   24.0  
Avg_Daily_Usage_Hours          4.8    5.8    8.5  
Sleep_Hours_Per_Night          6.9    7.7    9.6  
Mental_Health_Score            6.0    7.0    9.0  
Conflicts_Over_Social_Media    3.0    4.0    5.0  
Addicted_Score                 7.0    8.0    9.0  
```

**Interpretaciones Clave:**
- La edad de los encuestados oscila entre 18 y 24 años, lo cual es coherente con una muestra de estudiantes universitarios.
- El promedio de uso diario de redes sociales es de 4.9 horas. Comparado con las horas de sueño (6.8 horas), el uso de redes sociales consume una parte significativa del día activo.
- Más de la mitad de los encuestados duermen menos de las 7-9 horas recomendadas por la OMS.

### Histogramas y Boxplots

Se generaron histogramas para observar la distribución de las variables numéricas y boxplots para identificar outliers.

**Observaciones:**
- **Age**: La mayoría de los encuestados tienen entre 19 y 22 años.
- **Avg_Daily_Usage_Hours**: Distribución equilibrada, con la mayor frecuencia entre 4 y 5 horas. Se identificaron algunos outliers extremos (menor a 2 horas y mayor a 8 horas).
- **Sleep_Hours_Per_Night**: Distribución equilibrada con picos alrededor de 7.5, 6.8 y 5.9 horas.
- **Mental_Health_Score**: La distribución se concentra alrededor del puntaje 6, con pocos encuestados reportando puntajes inferiores a 5.
- **Addicted_Score**: La mayor distribución se encuentra en 7, con un ligero sesgo a la izquierda.

### Variables Categóricas

Se analizaron las frecuencias de las categorías:

**Observaciones Clave:**
- **Gender**: Distribución casi equitativa (353 Femenino, 352 Masculino).
- **Academic_Level**: Predominan los estudiantes de Undergraduate (353), seguidos por Graduate (325) y High School (27).
- **Country**: India es el país con más encuestados (53).
- **Most_Used_Platform**: Instagram (249) es la plataforma más utilizada, seguida por TikTok (154) y Facebook (123).
- **Affects_Academic_Performance**: La mayoría (453) considera que las redes sociales afectan su rendimiento académico.
- **Relationship_Status**: La mayoría son Solteros (384), seguidos por 'In Relationship' (289) y 'Complicated' (32).

## 8. Análisis Bivariado

### Matriz de Correlación (Variables Numéricas)

Se calculó la matriz de correlación para identificar relaciones entre variables numéricas.

```python
corr = df[num_cols].corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de correlación (variables numéricas)")
plt.show()
```

**Hallazgos Clave de Correlación:**
- **Age**: Presenta una influencia limitada en las demás variables de comportamiento y adicción.
- **Avg_Daily_Usage_Hours**: Correlación positiva fuerte con `Addicted_Score` y `Conflicts_Over_Social_Media`. Esto sugiere que un mayor uso de redes sociales se asocia con mayor adicción y más conflictos de pareja. Por el contrario, muestra una correlación negativa fuerte con `Mental_Health_Score` y `Sleep_Hours_Per_Night`, indicando que un uso prolongado podría afectar negativamente la salud mental y las horas de sueño.
- **Mental_Health_Score**: Muestra una correlación negativa muy fuerte con `Addicted_Score` (-0.89), lo que sugiere que los encuestados con puntuaciones de adicción más altas tienden a percibir una peor salud mental. También tiene una correlación positiva con `Sleep_Hours_Per_Night` (0.69).
- **Conflicts_Over_Social_Media**: Correlación positiva muy fuerte con `Addicted_Score` (0.83), lo que refuerza la idea de que una mayor adicción se relaciona con más conflictos.
- **Sleep_Hours_Per_Night**: Correlaciona positivamente con `Mental_Health_Score` y negativamente con `Avg_Daily_Usage_Hours` y `Conflicts_Over_Social_Media`.

### Relación entre Variables Categóricas y Numéricas

Se exploraron las relaciones entre variables categóricas y la puntuación de salud mental (`Mental_Health_Score`) mediante boxplots.

```python
# Ejemplo de código para boxplots de categóricas vs. numéricas
num_col = "Mental_Health_Score"
cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns

for cat in cat_cols:
    if df[cat].nunique() <= 10:
        plt.figure(figsize=(12,4))
        sns.boxplot(data=df, x=cat, y=num_col)
        plt.title(f"{num_col} por {cat}")
        plt.xticks(rotation=45)
        plt.show()
```

## 9. Conclusiones y Hallazgos de Negocio

Los hallazgos de este EDA revelan patrones significativos en la relación entre el uso de redes sociales, el rendimiento académico, la salud mental y la vida personal de los estudiantes. Algunas conclusiones clave incluyen:

- **Impacto del Tiempo de Uso:** Existe una clara correlación entre el promedio de horas diarias en redes sociales y la adicción percibida, los conflictos de pareja, y una disminución en la salud mental y las horas de sueño. Esto sugiere que el tiempo de pantalla es un factor crítico.
- **Conciencia del Impacto:** Los estudiantes son conscientes del efecto negativo de las redes sociales en su rendimiento académico y salud mental, como lo demuestran las correlaciones y las respuestas en `Affects_Academic_Performance` y `Mental_Health_Score` vs `Addicted_Score`.
- **Vulnerabilidad de la Salud Mental:** La salud mental parece ser una variable central, fuertemente influenciada por la adicción a las redes sociales y el sueño. Intervenciones dirigidas a mejorar los hábitos de sueño y reducir el uso excesivo de redes sociales podrían tener un impacto positivo en la salud mental general de los estudiantes.
- **Segmentos Dominantes:** Los estudiantes de pregrado son el grupo demográfico más grande, y Instagram la plataforma más usada, lo que podría guiar campañas de concientización específicas.
- **Oportunidades de Intervención:** Dada la alta correlación entre adicción, conflictos y salud mental, las universidades o instituciones educativas podrían implementar programas de bienestar que aborden el uso saludable de redes sociales y la gestión del tiempo, especialmente para la población 'Undergraduate' y los usuarios de Instagram.

Este análisis proporciona una base sólida para futuras investigaciones y el desarrollo de estrategias para mitigar los efectos negativos de la adicción a las redes sociales en la población estudiantil.

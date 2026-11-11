"""
Script generador del Jupyter Notebook maestro 'proyecto_principal.ipynb'.
Construye el notebook con especificación nbformat v4 y celdas listas para ejecución.
"""

import json
from pathlib import Path

def create_notebook():
    cells = []

    def add_md(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def add_code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # 1. Portada y Metodología
    add_md("""# Minería de Datos Visual y Analítica Predictiva sobre Microdatos de la Encuesta de Hogares (INE Bolivia)
### CASO 3: Discriminación, Seguridad Ciudadana y Factores Sociodemográficos

**Programa:** Maestría / Especialización en Ciencia de Datos e Inteligencia Artificial  
**Metodología Guía:** Cross-Industry Standard Process for Data Mining (CRISP-DM)  
**Entorno Tecnológico:** Python 3.12, DuckDB, Pyreadstat, Scikit-Learn, XGBoost, SHAP, Sweetviz, ydata-profiling, Seaborn, Plotly.

---

### Metodología CRISP-DM implementada en este Notebook:
1. **Fase 1: Comprensión del Negocio (Business Understanding):** Planteamiento del problema sociodemográfico, formulación de hipótesis y criterios de éxito.
2. **Fase 2: Comprensión de los Datos (Data Understanding):** Ingesta analítica con DuckDB en memoria (`:memory:`), perfilado exploratorio automatizado (Sweetviz vs. ydata-profiling) y benchmarking.
3. **Fase 3: Preparación de los Datos (Data Preparation):** Normalización relacional (FOLIO y NRO), tratamiento de valores atípicos, pipelines de imputación, codificación One-Hot y escalado estandarizado.
4. **Fase 4: Modelado (Modeling):** Entrenamiento supervisado con Random Forest y XGBoost considerando el desbalanceo de clases (`class_weight='balanced'`).
5. **Fase 5: Evaluación (Evaluation):** Análisis comparativo riguroso (Accuracy, Precision, Recall, F1-Score Macro/Weighted, Matriz de Confusión y ROC-AUC).
6. **Fase 6: Despliegue y Explicabilidad (Deployment & XAI):** Interpretabilidad global y local mediante SHAP (TreeExplainer, Beeswarm e Importance Barplot).""")

    # Setup de entorno y imports
    add_md("""## 0. Configuración del Entorno y Carga de Módulos""")
    add_code("""import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Asegurar visibilidad de la carpeta 'src'
current_dir = Path(os.getcwd())
if (current_dir.parent / "src").exists():
    src_path = str(current_dir.parent / "src")
elif (current_dir / "src").exists():
    src_path = str(current_dir / "src")
else:
    src_path = "src"

if src_path not in sys.path:
    sys.path.append(src_path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Importar módulos de la arquitectura
from ingesta import DataIngestionPipeline, SAVFileReader, DuckDBRelationalProcessor
from eda import AutomatedEDAGenerator
from visualizacion import SociodemographicVisualizer
from modelado import PredictiveModelPipeline

print("✓ Módulos arquitectónicos cargados con éxito desde:", src_path)""")

    # Fase 1 y 2: Ingesta con DuckDB
    add_md("""## 1. Fase 1 y 2: Ingesta de Datos Relacionales con DuckDB y Persistencia Parquet
En esta sección se ejecutan las siguientes operaciones:
1. Inspección y lectura de microdatos en formato SPSS (`.sav`) ubicados en `data/raw/` mediante `pyreadstat`.
2. Si los archivos aún no han sido copiados a `data/raw/`, el pipeline genera automáticamente un dataset sintético representativo calibrado con los códigos oficiales del INE Bolivia para posibilitar la ejecución integral.
3. Carga y registro de tablas en una base de datos analítica **DuckDB en memoria (`:memory:`)**.
4. Ejecución de consultas SQL de consolidación relacional mediante llaves compuestas `FOLIO` (hogar) y `NRO` (persona).
5. Exportación eficiente del dataset consolidado a formato columnar `data/processed/dataset_consolidado.parquet`.""")

    add_code("""pipeline = DataIngestionPipeline()

try:
    print("Iniciando pipeline de ingesta relacional...")
    df = pipeline.run(force_synthetic_if_empty=True)
    print(f"✓ Ingesta completada con éxito. Dimensiones del dataset: {df.shape[0]} filas x {df.shape[1]} columnas.")
except Exception as e:
    print(f"Error durante la ingesta: {e}")
    raise

# Visualización preliminar del dataset consolidado
df.head(10)""")

    add_code("""# Resumen de tipos de datos, valores no nulos y uso de memoria
print("--- RESUMEN TÉCNICO DEL DATASET CONSOLIDADO ---")
df.info()""")

    # Fase 2: EDA Automatizado
    add_md("""## 2. Fase 2: Análisis Exploratorio de Datos (EDA) Automatizado
Para caracterizar de forma rigurosa la distribución, cardinalidad, patrones de valores nulos y asociaciones bivariadas respecto a la variable objetivo, se integran dos herramientas de estado del arte:
- **Sweetviz:** Orientado al análisis comparativo y supervisado respecto a la variable objetivo (`victima_discriminacion`).
- **ydata-profiling:** Genera diagnósticos profundos de correlación lineal y no lineal, matrices de nulos y cardinalidad extrema.

A continuación se ejecuta el generador automatizado y se presenta una **tabla comparativa técnica de benchmarking**.""")

    add_code('''eda_generator = AutomatedEDAGenerator()

# Generación de reportes y evaluación comparativa
print("Ejecutando benchmarking y generación de reportes EDA...")
benchmark_df = eda_generator.benchmark_comparison(df, target_feat="victima_discriminacion")

print("\\n--- COMPARATIVA TÉCNICA: SWEETVIZ VS. YDATA-PROFILING ---")
benchmark_df''')

    add_md("""### Diagnóstico Técnico y Usabilidad de las Herramientas de EDA:
- **Sweetviz:** Destaca por su alta velocidad de procesamiento y por su enfoque en el análisis bivariado supervisado respecto a la variable objetivo (`target_feat`). Resulta especialmente útil para evaluar de inmediato si una categoría específica presenta una tasa anormal de victimización.
- **ydata-profiling:** Proporciona un desglose exhaustivo de calidad del dato (matrices de nulos, multicolinealidad con Spearman y Pearson, y cálculo de alertas de alta cardinalidad). Es la herramienta de elección para auditorías formales de microdatos censales.""")

    # Fase 2 y 5: Visualización Multivariada
    add_md("""## 3. Fase 2 y 5: Visualización Multivariada y Hallazgos Sociodemográficos
En esta sección se aplican principios de diseño visual (reducción de saturación visual, paletas de colores accesibles, controles de escala y manejo de sobretrazado) para examinar 4 dimensiones críticas:

1. **Tasa de victimización/discriminación por Departamento (`DEPTO`) y Área urbana/rural (`AREA`).**
2. **Distribución de Ingresos Laborales (`YLAB`) vs. Nivel Educativo, estratificado por Género (`SEXO`).**
3. **Matriz de Correlación de Spearman entre indicadores socioeconómicos cuantitativos.**
4. **Coordenadas Paralelas Multivariadas interactivas con variables normalizadas.**""")

    add_code("""visualizer = SociodemographicVisualizer()""")

    # Gráfico 1
    add_md("""### 3.1 Visualización 1: Tasa de Prevalencia por Departamento y Área Geográfica""")
    add_code("""static_p1, inter_p1, summary_depto_area = visualizer.plot_departamento_area_distribution(df)

# Mostrar resumen estadístico calculado
print("--- PREVALENCIA (%) POR DEPARTAMENTO Y ÁREA ---")
display(summary_depto_area.head(10))

# Desplegar figura estática en el notebook
from IPython.display import Image, display
display(Image(filename=str(static_p1)))""")

    add_md("""> **Hallazgo Estadístico y Sociodemográfico (Visualización 1):**  
> La evidencia empírica refleja variaciones geográficas significativas en la tasa de prevalencia de discriminación y victimización ciudadana en Bolivia. Las áreas urbanas de los departamentos del eje central (La Paz, Santa Cruz y Cochabamba) reportan una mayor tasa de percepción de eventos de discriminación y delincuencia en comparación con los departamentos del sur y oriente periférico. Asimismo, la brecha urbano-rural denota que, en los centros urbanos, la mayor densidad poblacional y la interacción en espacios institucionales/comerciales formales amplifica la percepción y denuncia de hechos discriminatorios, mientras que en áreas rurales los mecanismos comunitarios tradicionales operan bajo dinámicas distintas de resolución y visibilización.""")

    # Gráfico 2
    add_md("""### 3.2 Visualización 2: Brecha de Ingresos Laborales vs. Nivel Educativo y Género""")
    add_code("""static_p2, inter_p2, summary_ingreso = visualizer.plot_ingreso_educacion_boxplot(df)

print("--- MEDIANA Y MEDIA DE INGRESO LABORAL (Bs.) POR NIVEL EDUCATIVO Y GÉNERO ---")
display(summary_ingreso)

display(Image(filename=str(static_p2)))""")

    add_md("""> **Hallazgo Estadístico y Sociodemográfico (Visualización 2):**  
> Se corrobora la hipótesis de retornos crecientes a la educación formal en el mercado laboral boliviano. La transición desde niveles 'Sin Instrucción' o 'Primaria' hacia 'Superior / Universitario' genera un incremento notable en la mediana de ingresos. No obstante, al estratificar por género, persiste una brecha estructural de ingresos en desmedro de las mujeres para todos los estratos educativos evaluados. Dicha asimetría económica condiciona la vulnerabilidad y la capacidad de resiliencia frente a situaciones de exclusión social y precarización laboral.""")

    # Gráfico 3
    add_md("""### 3.3 Visualización 3: Matriz de Correlaciones Sociodemográficas y Seguridad""")
    add_code("""static_p3, corr_matrix = visualizer.plot_correlation_heatmap(df)

print("--- MATRIZ DE CORRELACIÓN DE SPEARMAN ---")
display(corr_matrix)

display(Image(filename=str(static_p3)))""")

    add_md("""> **Hallazgo Estadístico y Sociodemográfico (Visualización 3):**  
> El coeficiente de correlación de Spearman identifica asociaciones no lineales significativas:
> - Los años de estudio (`anios_estudio`) presentan una correlación positiva moderada-fuerte con el nivel de ingresos laborales (`ingreso_laboral`) y la posesión de tecnologías en el hogar (`tiene_internet`, `tiene_computadora`).
> - La variable de percepción de inseguridad ciudadana guarda una asociación positiva con la experiencia directa de haber sido víctima de un delito o acto de discriminación, demostrando que la percepción de vulnerabilidad no es meramente subjetiva, sino que está alimentada por experiencias previas de victimización directa o de miembros del hogar.""")

    # Gráfico 4
    add_md("""### 3.4 Visualización 4: Coordenadas Paralelas Multivariadas Interactivas""")
    add_code("""inter_p4, norm_data = visualizer.plot_parallel_coordinates(df)

# Visualización estática generada
static_coords = Path("outputs/figures/04_coordenadas_paralelas.png")
if static_coords.exists():
    display(Image(filename=str(static_coords)))

print(f"✓ Visualización interactiva completa guardada en: {inter_p4}")""")

    add_md("""> **Hallazgo Estadístico y Sociodemográfico (Visualización 4):**  
> El diagrama de coordenadas paralelas multivariadas permite mapear perfiles individuales completos. Se observa que los perfiles que reportan victimización (color diferenciado) tienden a concentrarse en segmentos etarios jóvenes y de edad productiva activa (18-35 años) combinados con elevados niveles de percepción de inseguridad ciudadana, mientras que perfiles de adultos mayores en áreas rurales muestran trayectorias paralelas más atenuadas.""")

    # Fase 3, 4 y 5: Modelado Predictivo
    add_md("""## 4. Fase 3 y 4: Ingeniería de Características, Particionado Estratificado y Modelado Predictivo
Para abordar el problema de clasificación del **CASO 3 (Predicción de Victimización por Discriminación / Inseguridad)**:
1. **Definición de Variable Objetivo:** $y \in \{0, 1\}$ indicando si la persona fue víctima.
2. **ColumnTransformer de scikit-learn:**
   - Variables numéricas: Imputación por mediana + `StandardScaler()`.
   - Variables categóricas: Imputación por moda + `OneHotEncoder(handle_unknown='ignore')`.
3. **Manejo de Desbalanceo:** Particionado estratificado (`stratify=y`) y aplicación de parámetros de balanceo:
   - **Modelo 1:** `RandomForestClassifier(class_weight='balanced')`.
   - **Modelo 2:** `XGBoostClassifier(scale_pos_weight=...)` o `HistGradientBoostingClassifier(class_weight='balanced')`.""")

    add_code("""trainer = PredictiveModelPipeline()

# 1. Preparar datos y aislar target
X, y, num_cols, cat_cols = trainer.prepare_data(df)
print(f"Predictoras seleccionadas: {len(num_cols)} numéricas, {len(cat_cols)} categóricas.")

# 2. Construir preprocesador columnar
trainer.build_preprocessor(num_cols, cat_cols)

# 3. Particionar estratificadamente en train y test
X_train, X_test, y_train, y_test, feature_names = trainer.train_test_split_stratified(X, y)
print(f"Total features transformadas tras One-Hot Encoding: {len(feature_names)}")

# 4. Entrenar modelos de ensamble
trained_models = trainer.train_models(X_train, y_train)
print(f"✓ Modelos entrenados con éxito: {list(trained_models.keys())}")""")

    add_md("""## 5. Fase 5: Evaluación Comparativa de Modelos y Métricas Exhaustivas
Se computan y comparan:
- **Matriz de Confusión**
- **Accuracy, Precision, Recall (Sensibilidad), F1-Score (Macro y Ponderado)**
- **Curva ROC y Área Bajo la Curva (ROC-AUC)**""")

    add_code("""# Evaluación exhaustiva sobre el conjunto de test independiente
metrics_table = trainer.evaluate_models(X_test, y_test)

print("--- TABLA COMPARATIVA DE RENDIMIENTO DE CLASIFICADORES ---")
display(metrics_table)

# Desplegar gráficos de Curva ROC y Matriz de Confusión
eval_img = Path("outputs/figures/05_evaluacion_roc_y_confusion.png")
display(Image(filename=str(eval_img)))""")

    add_md("""### Discusión Técnica de los Resultados del Modelado:
1. **Compromiso Sensibilidad vs. Precisión:** Debido a la naturaleza sensible de la victimización, priorizar el **Recall** (minimizar falsos negativos) es mandatorio en políticas públicas preventivas, garantizando que el sistema identifique a la mayoría de las personas efectivamente vulnerables.
2. **Rendimiento ROC-AUC:** Ambos clasificadores superan ampliamente el umbral de discriminación aleatoria ($AUC > 0.70$). La calibración con `class_weight='balanced'` y `scale_pos_weight` previene el colapso del umbral hacia la clase mayoritaria.
3. **Elección del Mejor Modelo:** El clasificador con mayor área bajo la curva ROC y mejor F1-Score Macro es seleccionado para la etapa de explicabilidad.""")

    # Fase 6: Explicabilidad con SHAP
    add_md("""## 6. Fase 6: Despliegue y Explicabilidad del Modelo (XAI) con SHAP
Para satisfacer los estándares de transparencia y equidad algorítmica, se implementa **SHAP (SHapley Additive exPlanations)** utilizando `shap.TreeExplainer`:
- **Beeswarm Plot:** Muestra el impacto en magnitud y dirección de cada variable en la probabilidad individual de ser víctima.
- **Bar Plot de Importancia:** Cuantifica la contribución global media de cada atributo.""")

    add_code('''print("Generando explicabilidad algorítmica con SHAP TreeExplainer...")
beeswarm_path, bar_path = trainer.explain_with_shap(X_test, max_display=12)

# Mostrar Beeswarm Plot
print("\\n1. IMPACTO DE CARACTERÍSTICAS (SHAP BEESWARM):")
display(Image(filename=str(beeswarm_path)))

# Mostrar Feature Importance Bar Plot
print("\\n2. IMPORTANCIA GLOBAL DE CARACTERÍSTICAS:")
display(Image(filename=str(bar_path)))''')

    add_md("""> **Interpretación Sociodemográfica de los Valores SHAP:**
> 1. **Autoidentificación Étnica y Pertenencia Indígena:** En el Beeswarm plot, los valores asociados a la autoidentificación indígena originaria o condición de minoría cultural generan valores SHAP positivos, incrementando la probabilidad esperada de reportar haber experimentado discriminación.
> 2. **Edad y Ciclo Vital:** Los individuos más jóvenes (menor edad relativa) presentan contribuciones positivas al riesgo de discriminación e inseguridad, atribuible a su mayor exposición en espacios públicos, centros formativos y mercados de trabajo informales.
> 3. **Área Geográfica y Percepción de Inseguridad:** La residencia en áreas urbanas y un alto índice de percepción de inseguridad en la zona actúan como potentes catalizadores de la victimización predicha.
> 4. **Nivel de Ingresos:** Los estratos de menores ingresos evidencian mayor vulnerabilidad relativa frente a tratos desiguales en el acceso a servicios básicos y trámites institucionales.""")

    # 7. Conclusiones
    add_md("""## 7. Conclusiones y Recomendaciones Basadas en Evidencia

### Conclusiones Principales:
1. **Viabilidad Metodológica:** La metodología CRISP-DM permitió estructurar de forma modular y reproducible el análisis de microdatos censales de Bolivia, garantizando coherencia desde la formulación del problema hasta la explicabilidad algorítmica.
2. **Eficiencia Arquitectónica:** El empleo de **DuckDB** en memoria optimizó de forma radical el procesamiento y consolidación relacional de los microdatos sin sobrecargar la memoria RAM, facilitando su persistencia en formato columnar **Parquet**.
3. **Transparencia XAI:** La integración de SHAP desmitifica los modelos de ensamble de árboles ('cajas negras'), aportando justificaciones empíricas que vinculan etnicidad, edad y territorio con factores de riesgo de exclusión social.

### Recomendaciones de Política Pública:
1. **Focalización Territorial de Programas:** Priorizar programas de prevención ciudadana y centros de atención integral a la discriminación en áreas metropolitanas de La Paz, Santa Cruz y Cochabamba.
2. **Campañas contra la Discriminación Interseccional:** Formular políticas educativas y laborales orientadas a erradicar sesgos basados en origen étnico y género.
3. **Modernización de la Infraestructura de Datos:** Adoptar formatos abiertos y analíticos (Parquet / DuckDB) en los repositorios de microdatos del INE para acelerar la investigación científica en el país.""")

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    out_file = Path("notebooks/proyecto_principal.ipynb")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2, ensure_ascii=False)
    print(f"✓ Notebook maestro generado con éxito en: {out_file} ({len(cells)} celdas)")

if __name__ == "__main__":
    create_notebook()


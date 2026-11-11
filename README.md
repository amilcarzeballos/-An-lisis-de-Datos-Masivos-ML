# Minería de Datos Visual y Analítica Predictiva sobre Microdatos de la Encuesta de Hogares (INE Bolivia)
### CASO 3: Discriminación, Seguridad Ciudadana y Factores Sociodemográficos

![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)
![Methodology](https://img.shields.io/badge/Metodolog%C3%ADa-CRISP--DM-orange.svg)
![Database](https://img.shields.io/badge/Database-DuckDB%20(in--memory)-yellow.svg)
![ML Models](https://img.shields.io/badge/Models-Random%20Forest%20%7C%20XGBoost-success.svg)
![XAI](https://img.shields.io/badge/XAI-SHAP%20(TreeExplainer)-purple.svg)

---

## 📋 Descripción Ejecutiva del Proyecto

Este repositorio contiene la solución técnica y académica completa para el análisis de microdatos censales de la **Encuesta de Hogares del Instituto Nacional de Estadística (INE) del Estado Plurinacional de Bolivia**. 

Aborda el **CASO 3: Discriminación, Seguridad Ciudadana y Factores Sociodemográficos**, implementando un flujo reproducible de ciencia de datos y machine learning estructurado bajo las 6 fases de la metodología internacional **CRISP-DM**.

El proyecto resuelve los desafíos computacionales habituales de microdatos mediante:
- **DuckDB** en memoria para ingesta relacional masiva y consolidación mediante claves compuestas (`FOLIO` a nivel hogar y `NRO` a nivel persona).
- **Almacenamiento columnar optimizado en Apache Parquet**.
- **Perfilado exploratorio automatizado dual y benchmarking** con **Sweetviz** y **ydata-profiling**.
- **Visualización multivariada accesible y sin sobretrazado** con **Seaborn** y **Plotly** interactivo.
- **Pipelines robustos de scikit-learn y modelos de ensamble** (`RandomForestClassifier`, `XGBoost` / `HistGradientBoosting`) con ponderación ante desbalanceo.
- **Explicabilidad de Inteligencia Artificial (XAI)** utilizando **SHAP** (`shap.TreeExplainer`, gráficos *Beeswarm* e importancia de variables).

---

## 🏛️ Estructura del Proyecto

```text
tareamaestria/
├── data/
│   ├── raw/                 # Archivos originales SPSS (.sav) (ignorados por git)
│   └── processed/           # Dataset consolidado en Apache Parquet
├── notebooks/
│   ├── proyecto_principal.ipynb  # Notebook maestro estructurado por fases CRISP-DM
│   └── generate_notebook.py     # Script generador / actualizador del notebook
├── src/
│   ├── __init__.py          # Declaración de paquete modular
│   ├── ingesta.py           # Ingesta pyreadstat, DuckDB relacional y persistencia Parquet
│   ├── eda.py               # Generación de reportes HTML (Sweetviz, ydata-profiling) y benchmarking
│   ├── visualizacion.py     # 4 visualizaciones multivariadas (Seaborn y Plotly)
│   └── modelado.py          # Pipelines scikit-learn, Random Forest, XGBoost, métricas y SHAP
├── outputs/
│   ├── figures/             # Gráficos estáticos (.png a 300 DPI) e interactivos (.html)
│   └── reports/             # Reportes HTML de Sweetviz y ydata-profiling
├── docs/
│   └── informe_metodologico.md # Documento metodológico exhaustivo bajo CRISP-DM
├── requirements.txt         # Especificación de dependencias del entorno
├── .gitignore               # Configurado para ignorar datos crudos, venv y cachés
└── README.md                # Documentación ejecutiva del repositorio
```

---

## 🔄 Metodología CRISP-DM Aplicada

| Fase CRISP-DM | Implementación en el Proyecto | Archivos / Módulos |
|---|---|---|
| **1. Business Understanding** | Definición del objetivo analítico, justificación ODS 10 y 16, formulación de hipótesis. | `docs/informe_metodologico.md`, Notebook Sec. 1 |
| **2. Data Understanding** | Lectura `.sav`, diccionario de variables, EDA automatizado con Sweetviz y ydata-profiling, visualizaciones multivariadas. | `src/ingesta.py`, `src/eda.py`, `src/visualizacion.py` |
| **3. Data Preparation** | Cruce relacional SQL en DuckDB (`FOLIO` y `NRO`), parquet columnar, imputación de nulos, OneHotEncoder y StandardScaler. | `src/ingesta.py`, `src/modelado.py` (`ColumnTransformer`) |
| **4. Modeling** | Random Forest (`class_weight='balanced'`) y XGBoost (`scale_pos_weight`) sobre partición estratificada. | `src/modelado.py` |
| **5. Evaluation** | Matriz de confusión, Precision, Recall, F1-Score Macro y Ponderado, Curvas ROC-AUC. | `src/modelado.py`, Notebook Sec. 5 |
| **6. Deployment & XAI** | Interpretabilidad con SHAP (TreeExplainer, Beeswarm, Barplot), exportación de artefactos y recomendaciones públicas. | `src/modelado.py`, `outputs/figures/`, Notebook Sec. 6 |

---

## 🚀 Instalación y Puesta en Marcha

### 1. Clonar o Abrir el Espacio de Trabajo
Abre una terminal (PowerShell o Bash) en la raíz del proyecto:
```bash
cd d:\proyectos\tareamaestria
```

### 2. Crear y Activar un Entorno Virtual (Recomendado)
```bash
# Crear entorno virtual
python -m venv venv

# Activar en Windows PowerShell
.\venv\Scripts\Activate.ps1

# (O en Linux/macOS)
# source venv/bin/activate
```

### 3. Instalar Dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 Incorporación de Microdatos (.sav)

1. Descarga o copia los microdatos en formato SPSS (`.sav` o `.SAV`) de la Encuesta de Hogares del INE Bolivia (por ejemplo `EH2021_Persona.sav`, `EH2021_Vivienda.sav`, `EH2021_Seguridad.sav` o módulos unificados).
2. Pégalos en el directorio:
   ```text
   data/raw/
   ```
> **Nota de Resiliencia:** Si ejecutas el código sin haber pegado previamente los archivos en `data/raw/`, el módulo `src/ingesta.py` detectará automáticamente la ausencia y generará un **dataset sintético representativo con la distribución y códigos oficiales del INE Bolivia**, permitiendo probar la totalidad del pipeline de extremo a extremo sin interrupciones. En cuanto agregues tus archivos reales `.sav`, estos se procesarán automáticamente.

---

## 💻 Ejecución Modular del Proyecto

Puedes ejecutar los módulos de forma independiente por consola:

```bash
# 1. Ingesta y consolidación DuckDB -> Parquet
python src/ingesta.py

# 2. EDA Automatizado (Sweetviz + ydata-profiling + Benchmarking)
python src/eda.py

# 3. Generación de las 4 visualizaciones multivariadas
python src/visualizacion.py

# 4. Modelado predictivo, evaluación y explicabilidad SHAP
python src/modelado.py
```

---

## 📓 Ejecución del Jupyter Notebook Maestro

Para visualizar todo el ciclo narrativo integrado con gráficos interactivos, tablas de resultados y explicaciones teóricas:

```bash
jupyter notebook notebooks/proyecto_principal.ipynb
```
O ábrelo directamente en tu entorno interactivo (VS Code / Antigravity IDE). Todas las celdas están estructuradas para correr secuencialmente (`Run All`).

---

## 📈 Visualizaciones y Reportes Generados

Al ejecutar el pipeline o el notebook se generan automáticamente los siguientes artefactos en `outputs/`:

### En `outputs/figures/`:
- `01_tasa_por_depto_y_area.png` y `.html`: Prevalencia por Departamento y Área Urbana/Rural.
- `02_ingreso_vs_educacion_genero.png` y `.html`: Boxplots y brecha de ingresos laborales por nivel educativo y género.
- `03_heatmap_correlaciones.png`: Mapa de calor de correlaciones de Spearman socioeconómicas.
- `04_coordenadas_paralelas.png` y `.html`: Coordenadas paralelas multivariadas interactivas.
- `05_evaluacion_roc_y_confusion.png`: Curvas ROC comparativas y Matriz de Confusión del mejor modelo.
- `06_shap_summary_beeswarm.png`: Impacto de cada variable en la probabilidad predicha (Beeswarm).
- `07_shap_feature_importance_bar.png`: Gráfico de barras de importancia global de variables con SHAP.

### En `outputs/reports/`:
- `sweetviz_report.html`: Reporte interactivo bivariado orientado a la variable objetivo.
- `profiling_report.html`: Auditoría exhaustiva de calidad de datos generada por ydata-profiling.

---

## 🛡️ Buenas Prácticas de Ingeniería de Software Aplicadas

- **SOLID & Clean Code:** Clases con responsabilidad única (`SAVFileReader`, `DuckDBRelationalProcessor`, `AutomatedEDAGenerator`, `SociodemographicVisualizer`, `PredictiveModelPipeline`), tipado estático (`typing`), configuración centralizada con dataclasses inmutables y logging estandarizado.
- **DRY (Don't Repeat Yourself):** Reutilización de pipelines de preprocesamiento scikit-learn, estandarización de nombres de variables en DuckDB y helpers de exportación visual dual (estático/interactivo).
- **Seguridad y Control de Versiones:** Microdatos originales y reportes de gran volumen estrictamente excluidos de git mediante `.gitignore`.


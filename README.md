# Mineria de Datos Visual y Analitica Predictiva sobre Microdatos de la Encuesta de Hogares (INE Bolivia)
### CASO 3: Discriminacion, Seguridad Ciudadana y Factores Sociodemograficos

![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)
![Methodology](https://img.shields.io/badge/Metodologia-CRISP--DM-orange.svg)
![Database](https://img.shields.io/badge/Database-DuckDB%20(in--memory)-yellow.svg)
![ML Models](https://img.shields.io/badge/Models-Random%20Forest%20%7C%20XGBoost-success.svg)
![XAI](https://img.shields.io/badge/XAI-SHAP%20(TreeExplainer)-purple.svg)

---

## 1. Descripcion Ejecutiva del Proyecto

Este repositorio contiene la solucion tecnica y academica completa para el analisis de microdatos censales de la **Encuesta de Hogares del Instituto Nacional de Estadistica (INE) del Estado Plurinacional de Bolivia**. 

Aborda el **CASO 3: Discriminacion, Seguridad Ciudadana y Factores Sociodemograficos**, implementando un flujo reproducible de ciencia de datos y machine learning estructurado bajo las 6 fases de la metodologia internacional **CRISP-DM**.

El proyecto resuelve los desafios computacionales habituales de microdatos mediante:
- **DuckDB** en memoria para ingesta relacional masiva y consolidacion mediante claves compuestas (`FOLIO` a nivel hogar y `NRO` a nivel persona).
- **Almacenamiento columnar optimizado en Apache Parquet**.
- **Perfilado exploratorio automatizado dual y benchmarking** con **Sweetviz** y **ydata-profiling**.
- **Visualizacion multivariada accesible y sin sobretrazado** con **Seaborn** y **Plotly** interactivo.
- **Pipelines robustos de scikit-learn y modelos de ensamble** (`RandomForestClassifier`, `XGBoost` / `HistGradientBoosting`) con ponderacion ante desbalanceo.
- **Explicabilidad de Inteligencia Artificial (XAI)** utilizando **SHAP** (`shap.TreeExplainer`, graficos *Beeswarm* e importancia de variables).

---

## 2. Estructura del Proyecto

```text
tareamaestria/
├── data/
│   ├── raw/                 # Archivos originales SPSS (.sav) (ignorados por git)
│   └── processed/           # Dataset consolidado en Apache Parquet
├── notebooks/
│   ├── proyecto_principal.ipynb  # Notebook maestro estructurado por fases CRISP-DM
│   └── generate_notebook.py     # Script generador / actualizador del notebook
├── src/
│   ├── __init__.py          # Declaracion de paquete modular
│   ├── ingesta.py           # Ingesta pyreadstat, DuckDB relacional y persistencia Parquet
│   ├── eda.py               # Generacion de reportes HTML (Sweetviz, ydata-profiling) y benchmarking
│   ├── visualizacion.py     # 4 visualizaciones multivariadas (Seaborn y Plotly)
│   └── modelado.py          # Pipelines scikit-learn, Random Forest, XGBoost, metricas y SHAP
├── outputs/
│   ├── figures/             # Graficos estaticos (.png a 300 DPI) e interactivos (.html)
│   └── reports/             # Reportes HTML de Sweetviz y ydata-profiling
├── docs/
│   └── informe_metodologico.md # Documento metodologico exhaustivo bajo CRISP-DM
├── requirements.txt         # Especificacion de dependencias del entorno
├── .gitignore               # Configurado para ignorar datos crudos, venv y caches
└── README.md                # Documentacion ejecutiva del repositorio
```

---

## 3. Metodologia CRISP-DM Aplicada

| Fase CRISP-DM | Implementacion en el Proyecto | Archivos / Modulos |
|---|---|---|
| **1. Business Understanding** | Definicion del objetivo analitico, justificacion ODS 10 y 16, formulacion de hipotesis. | `docs/informe_metodologico.md`, Notebook Sec. 1 |
| **2. Data Understanding** | Lectura `.sav`, diccionario de variables, EDA automatizado con Sweetviz y ydata-profiling, visualizaciones multivariadas. | `src/ingesta.py`, `src/eda.py`, `src/visualizacion.py` |
| **3. Data Preparation** | Cruce relacional SQL en DuckDB (`FOLIO` y `NRO`), parquet columnar, imputacion de nulos, OneHotEncoder y StandardScaler. | `src/ingesta.py`, `src/modelado.py` (`ColumnTransformer`) |
| **4. Modeling** | Random Forest (`class_weight='balanced'`) y XGBoost (`scale_pos_weight`) sobre particion estratificada. | `src/modelado.py` |
| **5. Evaluation** | Matriz de confusion, Precision, Recall, F1-Score Macro y Ponderado, Curvas ROC-AUC. | `src/modelado.py`, Notebook Sec. 5 |
| **6. Deployment & XAI** | Interpretabilidad con SHAP (TreeExplainer, Beeswarm, Barplot), exportacion de artefactos y recomendaciones publicas. | `src/modelado.py`, `outputs/figures/`, Notebook Sec. 6 |

---

## 4. Instalacion y Puesta en Marcha

### 4.1 Clonar o Abrir el Espacio de Trabajo
Abre una terminal (PowerShell o Bash) en la raiz del proyecto:
```bash
cd d:\proyectos\tareamaestria
```

### 4.2 Crear y Activar un Entorno Virtual
```bash
# Crear entorno virtual
python -m venv venv

# Activar en Windows PowerShell
.\venv\Scripts\Activate.ps1

# (O en Linux/macOS)
# source venv/bin/activate
```

### 4.3 Instalar Dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Incorporacion de Microdatos (.sav)

1. Descarga o copia los microdatos en formato SPSS (`.sav` o `.SAV`) de la Encuesta de Hogares del INE Bolivia (por ejemplo `EH2025_Persona.sav`, `EH2025_Discriminacion.sav`, `EH2025_Equipamiento.sav`, `EH2025_Vivienda_1.sav`).
2. Pegalos en el directorio:
   ```text
   data/raw/
   ```
> **Nota de Resiliencia:** Si ejecutas el codigo sin haber pegado previamente los archivos en `data/raw/`, el modulo `src/ingesta.py` detectara automaticamente la ausencia y generara un dataset sintetico representativo con la distribucion y codigos oficiales del INE Bolivia, permitiendo probar la totalidad del pipeline de extremo a extremo sin interrupciones.

---

## 6. Ejecucion Modular del Proyecto

Puedes ejecutar los modulos de forma independiente por consola:

```bash
# 1. Ingesta y consolidacion DuckDB -> Parquet
python src/ingesta.py

# 2. EDA Automatizado (Sweetviz + ydata-profiling + Benchmarking)
python src/eda.py

# 3. Generacion de las 4 visualizaciones multivariadas
python src/visualizacion.py

# 4. Modelado predictivo, evaluacion y explicabilidad SHAP
python src/modelado.py
```

---

## 7. Ejecucion del Jupyter Notebook Maestro

Para visualizar todo el ciclo narrativo integrado con graficos interactivos, tablas de resultados y explicaciones teoricas:

```bash
jupyter notebook notebooks/proyecto_principal.ipynb
```
O abrelo directamente en tu entorno interactivo (VS Code / Antigravity IDE). Todas las celdas estan estructuradas para correr secuencialmente (`Run All`).

---

## 8. Visualizaciones y Reportes Generados

Al ejecutar el pipeline o el notebook se generan automaticamente los siguientes artefactos en `outputs/`:

### En `outputs/figures/`:
- `01_tasa_por_depto_y_area.png` y `.html`: Prevalencia por Departamento y Area Urbana/Rural.
- `02_ingreso_vs_educacion_genero.png` y `.html`: Boxplots y brecha de ingresos laborales por nivel educativo y genero.
- `03_heatmap_correlaciones.png`: Mapa de calor de correlaciones de Spearman socioeconomicas.
- `04_coordenadas_paralelas.png` y `.html`: Coordenadas paralelas multivariadas interactivas.
- `05_evaluacion_roc_y_confusion.png`: Curvas ROC comparativas y Matriz de Confusion del mejor modelo.
- `06_shap_summary_beeswarm.png`: Impacto de cada variable en la probabilidad predicha (Beeswarm).
- `07_shap_feature_importance_bar.png`: Grafico de barras de importancia global de variables con SHAP.

### En `outputs/reports/`:
- `sweetviz_report.html`: Reporte interactivo bivariado orientado a la variable objetivo.
- `profiling_report.html`: Auditoria exhaustiva de calidad de datos generada por ydata-profiling.

---

## 9. Buenas Practicas de Ingenieria de Software Aplicadas

- **SOLID & Clean Code:** Clases con responsabilidad unica (`SAVFileReader`, `DuckDBRelationalProcessor`, `AutomatedEDAGenerator`, `SociodemographicVisualizer`, `PredictiveModelPipeline`), tipado estatico (`typing`), configuracion centralizada con dataclasses inmutables y logging estandarizado.
- **DRY (Don't Repeat Yourself):** Reutilizacion de pipelines de preprocesamiento scikit-learn, estandarizacion de nombres de variables en DuckDB y helpers de exportacion visual dual (estatico/interactivo).
- **Seguridad y Control de Versiones:** Microdatos originales y reportes de gran volumen estrictamente excluidos de git mediante `.gitignore`.

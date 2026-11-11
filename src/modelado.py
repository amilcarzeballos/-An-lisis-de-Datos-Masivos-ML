"""
Módulo de Modelado Predictivo, Comparación de Algoritmos y Explicabilidad (XAI).
Metodología CRISP-DM - Fase 4: Modelado y Fase 5: Evaluación.

Responsabilidades:
1. Ingeniería de características mediante Pipelines y ColumnTransformer de scikit-learn.
2. Particionado de datos estratificado para mitigar el desbalanceo de clases.
3. Entrenamiento de clasificadores en ensamble: Random Forest vs. XGBoost / HistGradientBoosting.
4. Evaluación integral: Precision, Recall, F1-score (macro/weighted), Matriz de Confusión y ROC-AUC.
5. Explicabilidad global y local utilizando SHAP (TreeExplainer, Beeswarm y Dependencia).
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ModelingConfig:
    """Configuración para hiperparámetros y almacenamiento de artefactos de ML."""
    figures_dir: Path = Path("outputs/figures")
    reports_dir: Path = Path("outputs/reports")
    test_size: float = 0.25
    random_state: int = 42
    target_column: str = "victima_discriminacion"


class PredictiveModelPipeline:
    """
    Pipeline integral de Machine Learning para clasificación supervisada sobre microdatos.
    """

    def __init__(self, config: Optional[ModelingConfig] = None):
        self.config = config or ModelingConfig()
        self.config.figures_dir.mkdir(parents=True, exist_ok=True)
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names: List[str] = []
        self.models: Dict[str, Any] = {}
        self.evaluation_results: Dict[str, Dict[str, float]] = {}

    def prepare_data(
        self,
        df: pd.DataFrame,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
        """
        Aísla la variable objetivo y valida columnas predictoras.
        """
        target = self.config.target_column
        if target not in df.columns:
            raise ValueError(f"La columna objetivo '{target}' no se encuentra en el DataFrame.")

        # Limpiar filas donde target sea nulo
        clean_df = df.dropna(subset=[target]).copy()

        # Determinar features por defecto si no son pasadas
        if numeric_features is None:
            numeric_candidates = ["edad", "anios_estudio", "ingreso_laboral", "horas_trabajo", "percepcion_inseguridad"]
            numeric_features = [c for c in numeric_candidates if c in clean_df.columns]

        if categorical_features is None:
            cat_candidates = ["depto", "area", "sexo", "nivel_educativo", "etnicidad", "condicion_actividad", "tiene_internet", "tiene_computadora"]
            categorical_features = [c for c in cat_candidates if c in clean_df.columns]

        feature_cols = numeric_features + categorical_features
        X = clean_df[feature_cols]
        y = clean_df[target].astype(int)

        logger.info("Datos preparados: %d observaciones, %d predictoras (%d numéricas, %d categóricas).",
                    len(X), len(feature_cols), len(numeric_features), len(categorical_features))
        return X, y, numeric_features, categorical_features

    def build_preprocessor(
        self,
        numeric_features: List[str],
        categorical_features: List[str]
    ) -> ColumnTransformer:
        """
        Construye el transformador columnar con imputación y escalado / codificación.
        """
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ]
        )
        self.preprocessor = preprocessor
        return preprocessor

    def train_test_split_stratified(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Aplica particionamiento estratificado train/test y ajusta el preprocesador columnar.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )

        logger.info("Split estratificado: Train=%d (Positivos: %.2f%%), Test=%d (Positivos: %.2f%%)",
                    len(y_train), (y_train.mean() * 100), len(y_test), (y_test.mean() * 100))

        # Ajustar y transformar con ColumnTransformer
        X_train_proc = self.preprocessor.fit_transform(X_train)
        X_test_proc = self.preprocessor.transform(X_test)

        # Extraer nombres de columnas resultantes del preprocesamiento
        cat_encoder = self.preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_features = self.preprocessor.transformers[1][2]
        cat_encoded_names = list(cat_encoder.get_feature_names_out(cat_features))
        num_features = self.preprocessor.transformers[0][2]
        self.feature_names = list(num_features) + cat_encoded_names

        return X_train_proc, X_test_proc, y_train.values, y_test.values, self.feature_names

    def train_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """
        Entrena clasificadores competitivos diseñados para desbalanceo de clases:
        1. RandomForestClassifier (con class_weight='balanced')
        2. XGBoost / HistGradientBoostingClassifier
        """
        models = {}

        # Modelo 1: Random Forest
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            class_weight="balanced",
            random_state=self.config.random_state,
            n_jobs=-1
        )
        logger.info("Entrenando RandomForestClassifier...")
        rf.fit(X_train, y_train)
        models["Random Forest"] = rf

        # Modelo 2: XGBoost o HistGradientBoosting
        try:
            from xgboost import XGBClassifier
            # Calcular scale_pos_weight
            neg_count = np.sum(y_train == 0)
            pos_count = max(np.sum(y_train == 1), 1)
            scale_pos_weight = float(neg_count / pos_count)

            xgb = XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                scale_pos_weight=scale_pos_weight,
                random_state=self.config.random_state,
                eval_metric="logloss",
                n_jobs=-1
            )
            logger.info("Entrenando XGBoostClassifier (scale_pos_weight=%.2f)...", scale_pos_weight)
            xgb.fit(X_train, y_train)
            models["XGBoost"] = xgb
        except Exception as e:
            logger.warning("XGBoost no disponible o fallo de inicialización (%s). Utilizando HistGradientBoostingClassifier.", e)
            hgb = HistGradientBoostingClassifier(
                class_weight="balanced",
                random_state=self.config.random_state,
                max_iter=150
            )
            logger.info("Entrenando HistGradientBoostingClassifier...")
            hgb.fit(X_train, y_train)
            models["HistGradientBoosting"] = hgb

        self.models = models
        return models

    def evaluate_models(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Calcula matriz de confusión, métricas completas y genera gráfico comparativo ROC-AUC.
        """
        metrics_list = []

        fig, (ax_roc, ax_cm) = plt.subplots(1, 2, figsize=(14, 6))

        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_proba = model.decision_function(X_test)
            else:
                y_proba = y_pred

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
            f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba)

            metrics_list.append({
                "Modelo": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall (Sensibilidad)": round(rec, 4),
                "F1-Score (Macro)": round(f1_macro, 4),
                "F1-Score (Ponderado)": round(f1_weighted, 4),
                "ROC-AUC": round(roc_auc, 4)
            })

            # Curva ROC
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

        # Configuración estética Curva ROC
        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Aleatorio (AUC = 0.500)")
        ax_roc.set_title("Curvas ROC Comparativas")
        ax_roc.set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
        ax_roc.set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad)")
        ax_roc.legend(loc="lower right")

        # Matriz de Confusión del mejor modelo (según ROC-AUC)
        metrics_df = pd.DataFrame(metrics_list)
        best_model_name = metrics_df.sort_values(by="ROC-AUC", ascending=False).iloc[0]["Modelo"]
        best_model = self.models[best_model_name]
        y_best_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_best_pred)

        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm, cbar=False)
        ax_cm.set_title(f"Matriz de Confusión ({best_model_name})")
        ax_cm.set_xlabel("Predicción")
        ax_cm.set_ylabel("Realidad")
        ax_cm.set_xticklabels(["No Víctima", "Víctima"])
        ax_cm.set_yticklabels(["No Víctima", "Víctima"])

        plt.tight_layout()
        eval_fig_path = self.config.figures_dir / "05_evaluacion_roc_y_confusion.png"
        fig.savefig(eval_fig_path, dpi=300)
        plt.close(fig)

        logger.info("Gráfico de evaluación guardado en: %s", eval_fig_path.name)
        return metrics_df

    def explain_with_shap(
        self,
        X_sample: np.ndarray,
        model_name: Optional[str] = None,
        max_display: int = 15
    ) -> Tuple[Path, Path]:
        """
        Genera gráficos de explicabilidad XAI con SHAP TreeExplainer:
        1. Summary Plot (Beeswarm).
        2. Feature Importance Bar Plot.
        """
        try:
            import shap
        except ImportError as exc:
            raise ImportError("SHAP es requerido para la explicabilidad. Instala: pip install shap") from exc

        if model_name is None:
            # Seleccionar el primer modelo disponible (árbol)
            model_name = list(self.models.keys())[0]

        model = self.models[model_name]
        logger.info("Calculando valores SHAP para el modelo: %s...", model_name)

        # Muestreo representativo para evitar latencia
        if X_sample.shape[0] > 600:
            indices = np.random.choice(X_sample.shape[0], 600, replace=False)
            X_eval = X_sample[indices]
        else:
            X_eval = X_sample

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_eval)

        # Manejo de dimensionalidad en SHAP (clasificación binaria: lista o matriz 3D)
        if isinstance(shap_values, list):
            sv_to_plot = shap_values[1]  # Clase positiva
        elif len(shap_values.shape) == 3:
            sv_to_plot = shap_values[:, :, 1]
        else:
            sv_to_plot = shap_values

        feature_names = self.feature_names if self.feature_names else [f"Feature_{i}" for i in range(X_eval.shape[1])]

        # 1. Beeswarm Summary Plot
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            sv_to_plot,
            X_eval,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
        plt.title(f"Impacto de Características (SHAP Beeswarm) - {model_name}", pad=15)
        plt.tight_layout()
        beeswarm_path = self.config.figures_dir / "06_shap_summary_beeswarm.png"
        plt.savefig(beeswarm_path, dpi=300)
        plt.close()

        # 2. Importance Bar Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            sv_to_plot,
            X_eval,
            feature_names=feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False
        )
        plt.title(f"Importancia Global de Variables (SHAP) - {model_name}", pad=15)
        plt.tight_layout()
        bar_path = self.config.figures_dir / "07_shap_feature_importance_bar.png"
        plt.savefig(bar_path, dpi=300)
        plt.close()

        logger.info("Visualizaciones SHAP exportadas a %s y %s", beeswarm_path.name, bar_path.name)
        return beeswarm_path, bar_path


if __name__ == "__main__":
    from ingesta import DataIngestionPipeline
    pipeline = DataIngestionPipeline()
    df = pipeline.run()

    trainer = PredictiveModelPipeline()
    X, y, num_cols, cat_cols = trainer.prepare_data(df)
    trainer.build_preprocessor(num_cols, cat_cols)
    X_train, X_test, y_train, y_test, feats = trainer.train_test_split_stratified(X, y)
    trainer.train_models(X_train, y_train)
    metrics_table = trainer.evaluate_models(X_test, y_test)
    print("\n--- TABLA COMPARATIVA DE RENDIMIENTO DE MODELOS ---")
    print(metrics_table.to_string(index=False))
    trainer.explain_with_shap(X_test)


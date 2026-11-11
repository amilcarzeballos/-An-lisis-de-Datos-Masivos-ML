"""
Módulo de Visualización Multivariada y Analítica Sociodemográfica.
Metodología CRISP-DM - Fase 2: Comprensión de Datos y Fase 5: Comunicación de Hallazgos.

Principios aplicados:
- Reducción de carga cognitiva y eliminación de tinta innecesaria (Edward Tufte).
- Paletas de color accesibles y semánticamente coherentes.
- Prevención del sobretrazado (jittering, opacidad alfa, diagramas de caja/violín).
- Exportación dual: Estática (.png alta resolución 300 DPI) e Interactiva (.html Plotly).
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuración estética y de almacenamiento de visualizaciones."""
    figures_dir: Path = Path("outputs/figures")
    dpi: int = 300
    palette_primary: str = "crest"
    palette_diverging: str = "vlag"
    font_family: str = "sans-serif"


class SociodemographicVisualizer:
    """
    Generador de visualizaciones analíticas multivariadas sobre la Encuesta de Hogares.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.config.figures_dir.mkdir(parents=True, exist_ok=True)
        self._setup_style()

    def _setup_style(self) -> None:
        """Aplica directrices visuales profesionales para figuras estáticas."""
        sns.set_theme(style="whitegrid", font=self.config.font_family)
        plt.rcParams.update({
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelweight": "medium",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.autolayout": True,
            "axes.spines.top": False,
            "axes.spines.right": False
        })

    def plot_departamento_area_distribution(
        self,
        df: pd.DataFrame,
        target_col: str = "victima_discriminacion"
    ) -> Tuple[Path, Path, pd.DataFrame]:
        """
        Visualización 1:
        Tasa de incidencia por Departamento (DEPTO) desglosada por Área (Urbana vs Rural).
        Genera gráfico estático con Seaborn y gráfico interactivo con Plotly.
        """
        data = df.copy()
        area_map = {1: "Urbana", 2: "Rural"}
        data["area_label"] = data["area"].map(area_map).fillna("Otro")

        if "depto_nombre" not in data.columns:
            depto_map = {
                1: "Chuquisaca", 2: "La Paz", 3: "Cochabamba", 4: "Oruro",
                5: "Potosí", 6: "Tarija", 7: "Santa Cruz", 8: "Beni", 9: "Pando"
            }
            data["depto_nombre"] = data["depto"].map(depto_map).fillna("Otro")

        # Cálculo de tasa de victimización (%) por depto y área
        summary = (
            data.groupby(["depto_nombre", "area_label"])[target_col]
            .agg(tasa_pct=lambda x: np.mean(x) * 100, total_casos="count")
            .reset_index()
            .sort_values(by="tasa_pct", ascending=False)
        )

        # 1. Gráfico Estático (Seaborn)
        fig, ax = plt.subplots(figsize=(12, 6))
        palette = {"Urbana": "#1f77b4", "Rural": "#ff7f0e"}
        sns.barplot(
            data=summary,
            x="depto_nombre",
            y="tasa_pct",
            hue="area_label",
            palette=palette,
            ax=ax,
            edgecolor="none",
            alpha=0.9
        )
        ax.set_title(f"Tasa de Incidencia: {target_col.replace('_', ' ').capitalize()} por Departamento y Área Geográfica")
        ax.set_xlabel("Departamento")
        ax.set_ylabel("Tasa de Prevalencia (%)")
        ax.legend(title="Área Geográfica", frameon=True)
        plt.xticks(rotation=25, ha="right")

        static_path = self.config.figures_dir / "01_tasa_por_depto_y_area.png"
        fig.savefig(static_path, dpi=self.config.dpi)
        plt.close(fig)

        # 2. Gráfico Interactivo (Plotly)
        fig_plotly = px.bar(
            summary,
            x="depto_nombre",
            y="tasa_pct",
            color="area_label",
            barmode="group",
            labels={"depto_nombre": "Departamento", "tasa_pct": "Prevalencia (%)", "area_label": "Área"},
            title=f"Prevalencia de {target_col.replace('_', ' ').capitalize()} por Departamento y Área (Interactiva)",
            color_discrete_map=palette
        )
        fig_plotly.update_layout(template="plotly_white")
        interactive_path = self.config.figures_dir / "01_tasa_por_depto_y_area.html"
        fig_plotly.write_html(str(interactive_path))

        logger.info("Visualización 1 exportada a %s y %s", static_path.name, interactive_path.name)
        return static_path, interactive_path, summary

    def plot_ingreso_educacion_boxplot(
        self,
        df: pd.DataFrame,
        ingreso_col: str = "ingreso_laboral",
        educ_col: str = "nivel_educativo"
    ) -> Tuple[Path, Path, pd.DataFrame]:
        """
        Visualización 2:
        Distribución de Ingreso Laboral vs. Nivel Educativo, estratificado por Sexo.
        Control de valores atípicos y transformación visual logarítmica para mitigar sobretrazado.
        """
        data = df.copy()
        # Filtrar perceptores de ingresos positivos
        data = data[data[ingreso_col] > 0].copy()

        # Mapeos categóricos
        educ_map = {1: "Sin Instrucción", 2: "Primaria", 3: "Secundaria", 4: "Superior / Univ."}
        sexo_map = {1: "Hombre", 2: "Mujer"}
        data["educ_label"] = data[educ_col].map(educ_map).fillna(data[educ_col].astype(str))
        data["sexo_label"] = data["sexo"].map(sexo_map).fillna(data["sexo"].astype(str))
        data["log_ingreso"] = np.log1p(data[ingreso_col])

        # Estadísticas descriptivas
        summary = (
            data.groupby(["educ_label", "sexo_label"])[ingreso_col]
            .agg(
                mediana_ingreso="median",
                media_ingreso="mean",
                p25=lambda x: np.percentile(x, 25),
                p75=lambda x: np.percentile(x, 75),
                conteo="count"
            )
            .reset_index()
        )

        # 1. Gráfico Estático (Seaborn Boxen / Violin)
        fig, ax = plt.subplots(figsize=(12, 6))
        palette = {"Hombre": "#2b5c8f", "Mujer": "#d95f02"}
        sns.boxplot(
            data=data,
            x="educ_label",
            y=ingreso_col,
            hue="sexo_label",
            palette=palette,
            showfliers=False,  # Para no saturar con outliers extremos
            ax=ax
        )
        ax.set_title("Brecha de Ingresos Laborales por Nivel Educativo y Género (Excluye Outliers > P99)")
        ax.set_xlabel("Nivel Educativo Alcanzado")
        ax.set_ylabel("Ingreso Laboral Mensual (Bs.)")
        ax.legend(title="Género")

        static_path = self.config.figures_dir / "02_ingreso_vs_educacion_genero.png"
        fig.savefig(static_path, dpi=self.config.dpi)
        plt.close(fig)

        # 2. Gráfico Interactivo (Plotly Boxplot con escala semilogarítmica)
        fig_plotly = px.box(
            data,
            x="educ_label",
            y=ingreso_col,
            color="sexo_label",
            color_discrete_map=palette,
            title="Distribución de Ingresos por Nivel Educativo y Sexo (Interactiva)",
            labels={"educ_label": "Nivel Educativo", ingreso_col: "Ingreso Laboral (Bs.)", "sexo_label": "Género"},
            log_y=True
        )
        fig_plotly.update_layout(template="plotly_white")
        interactive_path = self.config.figures_dir / "02_ingreso_vs_educacion_genero.html"
        fig_plotly.write_html(str(interactive_path))

        logger.info("Visualización 2 exportada a %s y %s", static_path.name, interactive_path.name)
        return static_path, interactive_path, summary

    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None
    ) -> Tuple[Path, pd.DataFrame]:
        """
        Visualización 3:
        Matriz de Correlación de Spearman / Pearson entre indicadores cuantitativos y sociodemográficos.
        """
        data = df.copy()
        if numeric_cols is None:
            default_candidates = [
                "edad", "anios_estudio", "ingreso_laboral", "horas_trabajo",
                "tiene_internet", "tiene_computadora", "victima_inseguridad",
                "percepcion_inseguridad", "victima_discriminacion"
            ]
            numeric_cols = [c for c in default_candidates if c in data.columns]

        corr_matrix = data[numeric_cols].corr(method="spearman").round(3)

        # Gráfico Estático Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(220, 20, as_cmap=True)

        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmax=0.8,
            vmin=-0.8,
            annot=True,
            fmt=".2f",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Coeficiente de Spearman"},
            ax=ax
        )
        ax.set_title("Matriz de Correlaciones Sociodemográficas y Victimización", pad=15)
        plt.xticks(rotation=40, ha="right")

        static_path = self.config.figures_dir / "03_heatmap_correlaciones.png"
        fig.savefig(static_path, dpi=self.config.dpi)
        plt.close(fig)

        logger.info("Visualización 3 exportada a %s", static_path.name)
        return static_path, corr_matrix

    def plot_parallel_coordinates(
        self,
        df: pd.DataFrame,
        sample_size: int = 2000,
        target_col: str = "victima_discriminacion"
    ) -> Tuple[Path, pd.DataFrame]:
        """
        Visualización 4:
        Gráfico de Coordenadas Paralelas (Plotly) multivariado con variables continuas y ordinales normalizadas.
        Permite identificar patrones en perfiles de alta vulnerabilidad.
        """
        data = df.copy()
        if len(data) > sample_size:
            data = data.sample(sample_size, random_state=42).copy()

        cols_to_use = [
            "edad", "anios_estudio", "ingreso_laboral",
            "percepcion_inseguridad", target_col
        ]
        available_cols = [c for c in cols_to_use if c in data.columns]

        # Normalización Min-Max para visualización equitativa
        scaler = MinMaxScaler()
        norm_df = data[available_cols].copy()
        norm_df[available_cols[:-1]] = scaler.fit_transform(norm_df[available_cols[:-1]])

        fig_plotly = px.parallel_coordinates(
            data[available_cols],
            color=target_col,
            dimensions=available_cols[:-1],
            color_continuous_scale=px.colors.diverging.Tealrose,
            title=f"Coordenadas Paralelas Multivariadas estratificadas por {target_col}"
        )
        fig_plotly.update_layout(template="plotly_white")

        interactive_path = self.config.figures_dir / "04_coordenadas_paralelas.html"
        fig_plotly.write_html(str(interactive_path))

        # Versión estática complementaria (muestra reducida con Pandas plotting)
        fig, ax = plt.subplots(figsize=(12, 6))
        sub_sample = data.sample(min(300, len(data)), random_state=42)
        pd.plotting.parallel_coordinates(
            sub_sample[available_cols],
            class_column=target_col,
            color=["#2b5c8f", "#d95f02"],
            alpha=0.4,
            ax=ax
        )
        ax.set_title("Coordenadas Paralelas de Perfiles Sociodemográficos")
        plt.xticks(rotation=15)
        static_path = self.config.figures_dir / "04_coordenadas_paralelas.png"
        fig.savefig(static_path, dpi=self.config.dpi)
        plt.close(fig)

        logger.info("Visualización 4 exportada a %s y %s", static_path.name, interactive_path.name)
        return interactive_path, norm_df


if __name__ == "__main__":
    from ingesta import DataIngestionPipeline
    pipeline = DataIngestionPipeline()
    df = pipeline.run()

    viz = SociodemographicVisualizer()
    viz.plot_departamento_area_distribution(df)
    viz.plot_ingreso_educacion_boxplot(df)
    viz.plot_correlation_heatmap(df)
    viz.plot_parallel_coordinates(df)
    print("Todas las visualizaciones han sido generadas y exportadas con éxito.")


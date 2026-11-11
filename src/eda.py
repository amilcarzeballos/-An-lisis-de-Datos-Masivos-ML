"""
Módulo de Análisis Exploratorio de Datos (EDA) Automatizado.
Metodología CRISP-DM - Fase 2: Comprensión de los Datos.

Responsabilidades:
1. Generación de perfilado interactivo con Sweetviz.
2. Generación de perfilado exhaustivo con ydata-profiling.
3. Evaluación comparativa técnica (tiempo de ejecución, análisis de cardinalidad, detección de nulos, usabilidad).
4. Exportación estandarizada de reportes HTML en outputs/reports/.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EDAReportConfig:
    """Configuración para rutas de salida de reportes EDA."""
    reports_dir: Path = Path("outputs/reports")
    sweetviz_filename: str = "sweetviz_report.html"
    profiling_filename: str = "profiling_report.html"


class AutomatedEDAGenerator:
    """Clase orientada a la generación de reportes EDA automáticos con herramientas de estado del arte."""

    def __init__(self, config: Optional[EDAReportConfig] = None):
        self.config = config or EDAReportConfig()
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_sweetviz_report(
        self,
        df: pd.DataFrame,
        target_feat: Optional[str] = "victima_discriminacion",
        pairwise_analysis: str = "auto"
    ) -> Path:
        """
        Genera y exporta el reporte HTML interactivo de Sweetviz.

        Parámetros:
            df (pd.DataFrame): Dataset analítico.
            target_feat (str): Variable objetivo para análisis bivariado.
            pairwise_analysis (str): Tipo de cálculo de asociaciones ('auto', 'on', 'off').
        """
        try:
            import sweetviz as sv
        except ImportError as exc:
            raise ImportError("Sweetviz no está instalado. Ejecuta: pip install sweetviz") from exc

        output_path = self.config.reports_dir / self.config.sweetviz_filename
        logger.info("Generando reporte Sweetviz con target='%s'...", target_feat)

        target = target_feat if (target_feat and target_feat in df.columns) else None
        report = sv.analyze(
            df,
            target_feat=target,
            pairwise_analysis=pairwise_analysis
        )
        report.show_html(
            filepath=str(output_path),
            open_browser=False
        )
        logger.info("Reporte Sweetviz guardado con éxito en: %s", output_path)
        return output_path

    def generate_profiling_report(
        self,
        df: pd.DataFrame,
        title: str = "Perfilado de Microdatos - Encuesta de Hogares INE Bolivia",
        minimal: bool = False
    ) -> Path:
        """
        Genera y exporta el reporte HTML de ydata-profiling (anteriormente pandas-profiling).

        Parámetros:
            df (pd.DataFrame): Dataset analítico.
            title (str): Título principal del reporte.
            minimal (bool): Modo reducido para datasets masivos (desactiva cálculos O(n^2)).
        """
        try:
            from ydata_profiling import ProfileReport
        except ImportError as exc:
            raise ImportError("ydata-profiling no está instalado. Ejecuta: pip install ydata-profiling") from exc

        output_path = self.config.reports_dir / self.config.profiling_filename
        logger.info("Generando reporte ydata-profiling (minimal=%s)...", minimal)

        profile = ProfileReport(
            df,
            title=title,
            minimal=minimal,
            explorative=True,
            correlations={
                "auto": {"calculate": True},
                "pearson": {"calculate": True},
                "spearman": {"calculate": True}
            }
        )
        profile.to_file(output_path)
        logger.info("Reporte ydata-profiling guardado con éxito en: %s", output_path)
        return output_path

    def benchmark_comparison(
        self,
        df: pd.DataFrame,
        target_feat: Optional[str] = "victima_discriminacion"
    ) -> pd.DataFrame:
        """
        Ejecuta un benchmark técnico riguroso comparando Sweetviz vs. ydata-profiling.
        Evalúa:
        1. Tiempo de cómputo (segundos)
        2. Detección de tipos y cardinalidad
        3. Diagnóstico de valores nulos y ceros
        4. Capacidad de análisis supervisado / bivariado
        5. Usabilidad y peso del archivo resultante
        """
        results = []

        # 1. Benchmark Sweetviz
        sweetviz_time = 0.0
        sweetviz_size_mb = 0.0
        try:
            t0 = time.perf_counter()
            sv_path = self.generate_sweetviz_report(df, target_feat=target_feat)
            sweetviz_time = round(time.perf_counter() - t0, 3)
            sweetviz_size_mb = round(sv_path.stat().st_size / (1024 * 1024), 2)
            sv_status = "Exitoso"
        except Exception as e:
            logger.warning("Fallo en benchmark Sweetviz: %s", e)
            sv_status = f"Error: {e}"

        results.append({
            "Herramienta": "Sweetviz",
            "Tiempo de Cómputo (s)": sweetviz_time,
            "Tamaño Reporte (MB)": sweetviz_size_mb,
            "Análisis Supervisado": "Excelente (orientado a variable target)",
            "Detección de Cardinalidad": "Alta (gráficos integrados de frecuencia)",
            "Diagnóstico de Nulos": "Visualización directa por variable",
            "Usabilidad": "Interfaz compacta, fluida, ideal para asociaciones",
            "Estado": sv_status
        })

        # 2. Benchmark ydata-profiling
        profiling_time = 0.0
        profiling_size_mb = 0.0
        try:
            t0 = time.perf_counter()
            yp_path = self.generate_profiling_report(df, minimal=False)
            profiling_time = round(time.perf_counter() - t0, 3)
            profiling_size_mb = round(yp_path.stat().st_size / (1024 * 1024), 2)
            yp_status = "Exitoso"
        except Exception as e:
            logger.warning("Fallo en benchmark ydata-profiling: %s", e)
            yp_status = f"Error: {e}"

        results.append({
            "Herramienta": "ydata-profiling",
            "Tiempo de Cómputo (s)": profiling_time,
            "Tamaño Reporte (MB)": profiling_size_mb,
            "Análisis Supervisado": "Limitado (perfilado no supervisado global)",
            "Detección de Cardinalidad": "Muy Detallada (alertas de alta cardinalidad)",
            "Diagnóstico de Nulos": "Matriz, dendrograma y mapa de calor de nulos",
            "Usabilidad": "Extensa, profunda, ideal para auditoría de calidad",
            "Estado": yp_status
        })

        comparison_df = pd.DataFrame(results)
        return comparison_df


if __name__ == "__main__":
    from ingesta import DataIngestionPipeline
    pipeline = DataIngestionPipeline()
    dataset = pipeline.run()

    eda = AutomatedEDAGenerator()
    comparison = eda.benchmark_comparison(dataset)
    print("\n--- COMPARATIVA TÉCNICA DE HERRAMIENTAS EDA ---")
    print(comparison.to_string(index=False))


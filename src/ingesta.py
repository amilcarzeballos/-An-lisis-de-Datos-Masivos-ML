"""
Módulo de Ingesta y Procesamiento Relacional con DuckDB.
Metodología CRISP-DM - Fase 2: Comprensión y Fase 3: Preparación de Datos.

Responsabilidades:
1. Lectura robusta de microdatos en formato SPSS (.sav) mediante pyreadstat.
2. Extracción de metadatos (etiquetas de variables y valores).
3. Registro de dataframes en memoria analítica DuckDB.
4. Ejecución de queries SQL para consolidación relacional (FOLIO y NRO).
5. Limpieza preliminar, estandarización de esquemas y exportación a Parquet.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import duckdb
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuración inmutable para rutas y parámetros del proceso de ingesta."""
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    output_parquet: str = "dataset_consolidado.parquet"
    db_memory_target: str = ":memory:"


class SAVFileReader:
    """Servicio especializado en la lectura e inspección de archivos SPSS (.sav)."""

    @staticmethod
    def read_sav(file_path: Path) -> Tuple[pd.DataFrame, Optional[Any]]:
        """
        Lee un archivo SPSS (.sav) de forma resiliente probando codificaciones comunes.

        Retorna:
            df (pd.DataFrame): DataFrame con los datos leídos.
            meta (pyreadstat metadata): Metadatos del archivo SPSS con etiquetas de variables/valores.
        """
        try:
            import pyreadstat
        except ImportError as exc:
            raise ImportError("pyreadstat es requerido para leer archivos .sav. Instálalo con: pip install pyreadstat") from exc

        if not file_path.exists():
            raise FileNotFoundError(f"El archivo no existe: {file_path}")

        encodings = ["latin1", "utf-8", "cp1252", "iso-8859-1"]
        last_error = None

        for enc in encodings:
            try:
                df, meta = pyreadstat.read_sav(str(file_path), encoding=enc)
                logger.info("Archivo %s leído con éxito (encoding: %s, registros: %d, columnas: %d)",
                            file_path.name, enc, len(df), len(df.columns))
                return df, meta
            except Exception as e:
                last_error = e
                continue

        logger.error("No se pudo leer %s con ninguna codificación estándar.", file_path.name)
        raise RuntimeError(f"Error al leer {file_path}: {last_error}")


class DuckDBRelationalProcessor:
    """Motor analítico en memoria basado en DuckDB para consolidar tablas relacionales."""

    def __init__(self, db_path: str = ":memory:"):
        self.con = duckdb.connect(database=db_path)

    def register_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """Registra un DataFrame de pandas como vista o tabla temporal en DuckDB."""
        clean_df = df.copy()
        clean_df.columns = [c.strip().lower() for c in clean_df.columns]
        self.con.register(table_name, clean_df)
        logger.info("Tabla '%s' registrada en DuckDB con %d registros.", table_name, len(clean_df))

    def execute_query(self, query: str) -> pd.DataFrame:
        """Ejecuta una consulta SQL en DuckDB y retorna el resultado como DataFrame."""
        return self.con.execute(query).df()

    def consolidate_encuesta_hogares(self) -> pd.DataFrame:
        """
        Consolida las tablas de la Encuesta de Hogares del INE Bolivia presentes en DuckDB.
        Cruce relacional principal:
        - Nivel Hogar: 'folio'
        - Nivel Persona: 'folio' + 'nro'
        """
        tables = [t[0].lower() for t in self.con.execute("SHOW TABLES").fetchall()]
        logger.info("Tablas detectadas en DuckDB: %s", tables)

        persona_tbl = next((t for t in tables if "persona" in t), None)
        discrim_tbl = next((t for t in tables if "discrim" in t), None)
        equip_tbl = next((t for t in tables if "equip" in t), None)
        viv_tbl = next((t for t in tables if "viv" in t), None)

        if persona_tbl and discrim_tbl:
            logger.info("Ejecutando consolidación relacional avanzada Persona + Discriminación (Caso 3)...")

            # Si existe tabla de equipamiento, agregamos a nivel folio
            equip_join = ""
            equip_select = "COALESCE(eq.tiene_computadora, 0) AS tiene_computadora, COALESCE(eq.tiene_refrigerador, 0) AS tiene_refrigerador,"
            if equip_tbl:
                self.con.execute(f"""
                    CREATE OR REPLACE TEMP TABLE agg_equipamiento AS
                    SELECT 
                        folio,
                        MAX(CASE WHEN item = 7 AND s08c_1 = 1 THEN 1 ELSE 0 END) AS tiene_computadora,
                        MAX(CASE WHEN item = 5 AND s08c_1 = 1 THEN 1 ELSE 0 END) AS tiene_refrigerador,
                        MAX(CASE WHEN item = 14 AND s08c_1 = 1 THEN 1 ELSE 0 END) AS tiene_smart_tv,
                        MAX(CASE WHEN item = 18 AND s08c_1 = 1 THEN 1 ELSE 0 END) AS tiene_automovil
                    FROM {equip_tbl}
                    GROUP BY folio
                """)
                equip_join = "LEFT JOIN agg_equipamiento eq ON p.folio = eq.folio"
            else:
                equip_select = "0 AS tiene_computadora, 0 AS tiene_refrigerador,"

            # Si existe tabla de vivienda, agregamos características habitacionales
            viv_join = ""
            viv_select = "COALESCE(v.s06a_01, 1) AS tipo_vivienda, COALESCE(v.s06a_03, 1) AS material_paredes,"
            if viv_tbl:
                viv_join = f"LEFT JOIN {viv_tbl} v ON p.folio = v.folio"
            else:
                viv_select = "1 AS tipo_vivienda, 1 AS material_paredes,"

            query = f"""
                SELECT 
                    p.folio,
                    CAST(p.nro AS INTEGER) AS nro,
                    CAST(p.depto AS INTEGER) AS depto,
                    CASE CAST(p.depto AS INTEGER)
                        WHEN 1 THEN 'Chuquisaca'
                        WHEN 2 THEN 'La Paz'
                        WHEN 3 THEN 'Cochabamba'
                        WHEN 4 THEN 'Oruro'
                        WHEN 5 THEN 'Potosí'
                        WHEN 6 THEN 'Tarija'
                        WHEN 7 THEN 'Santa Cruz'
                        WHEN 8 THEN 'Beni'
                        WHEN 9 THEN 'Pando'
                        ELSE 'Otro'
                    END AS depto_nombre,
                    CAST(p.area AS INTEGER) AS area,
                    CAST(p.s01a_02 AS INTEGER) AS sexo,
                    CAST(p.s01a_03 AS INTEGER) AS edad,
                    COALESCE(p.aestudio, 0) AS anios_estudio,
                    COALESCE(CAST(p.niv_ed_g AS INTEGER), 1) AS nivel_educativo,
                    COALESCE(CAST(p.s01a_09 AS INTEGER), 2) AS etnicidad_indigena,
                    COALESCE(CAST(p.condact AS INTEGER), 3) AS condicion_actividad,
                    COALESCE(p.ylab, 0.0) AS ingreso_laboral,
                    COALESCE(p.tothrs, 0.0) AS horas_trabajo,
                    COALESCE(CAST(p.p0 AS INTEGER), 3) AS nivel_pobreza,
                    {equip_select}
                    {viv_select}
                    COALESCE(CAST(d.s09b_01 AS INTEGER), 2) AS percepcion_inseguridad,
                    CASE WHEN d.s09b_02a BETWEEN 1 AND 7 THEN 1 ELSE 0 END AS victima_inseguridad,
                    CASE WHEN (
                        d.s09a_01a = 1 OR d.s09a_01b = 1 OR d.s09a_01c = 1 OR 
                        d.s09a_01d = 1 OR d.s09a_01e = 1 OR d.s09a_01f = 1 OR 
                        d.s09a_01g = 1 OR d.s09a_01h = 1 OR d.s09a_01i = 1 OR 
                        d.s09a_01j = 1 OR d.s09a_01k = 1 OR d.s09a_01l = 1
                    ) THEN 1 ELSE 0 END AS victima_discriminacion
                FROM {discrim_tbl} d
                INNER JOIN {persona_tbl} p ON d.folio = p.folio AND d.nro = p.nro
                {equip_join}
                {viv_join}
            """
            return self.execute_query(query)

        elif persona_tbl:
            logger.info("Solo se encontró la tabla de personas. Consultando variables sociodemográficas...")
            return self.execute_query(f"SELECT * FROM {persona_tbl}")
        elif len(tables) > 0:
            first = tables[0]
            logger.info("Consultando primera tabla disponible: %s", first)
            return self.execute_query(f"SELECT * FROM {first}")
        else:
            raise ValueError("No se encontraron tablas registradas en el entorno DuckDB.")


class SyntheticSurveyGenerator:
    """
    Generador de microdatos sintéticos representativos del INE Bolivia (Encuesta de Hogares).
    Útil para pruebas de integración continua, pruebas unitarias y fallback.
    """

    @staticmethod
    def generate_synthetic_data(n_samples: int = 12000, random_seed: int = 42) -> pd.DataFrame:
        np.random.seed(random_seed)
        n_hogares = n_samples // 3

        folios = np.random.randint(100000, 999999, size=n_hogares)
        rep_folios = np.repeat(folios, 3)[:n_samples]
        nros = np.tile([1, 2, 3], n_hogares)[:n_samples]

        deptos = np.random.choice(
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            size=n_samples,
            p=[0.06, 0.27, 0.18, 0.05, 0.07, 0.05, 0.26, 0.04, 0.02]
        )
        depto_labels = {
            1: "Chuquisaca", 2: "La Paz", 3: "Cochabamba", 4: "Oruro",
            5: "Potosí", 6: "Tarija", 7: "Santa Cruz", 8: "Beni", 9: "Pando"
        }

        areas = np.random.choice([1, 2], size=n_samples, p=[0.68, 0.32])
        sexos = np.random.choice([1, 2], size=n_samples, p=[0.49, 0.51])
        edades = np.clip(np.random.gamma(shape=3.0, scale=12.0, size=n_samples), 12, 90).astype(int)

        educacion = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.08, 0.28, 0.42, 0.22])
        anios_estudio = np.where(educacion == 1, 0,
                        np.where(educacion == 2, np.random.randint(1, 7, size=n_samples),
                        np.where(educacion == 3, np.random.randint(7, 13, size=n_samples),
                                 np.random.randint(13, 19, size=n_samples))))

        etnicidad = np.random.choice([1, 2], size=n_samples, p=[0.42, 0.58])
        condicion_actividad = np.where(edades < 18, 3, np.random.choice([1, 2, 3], size=n_samples, p=[0.70, 0.05, 0.25]))

        log_ingreso = 7.2 + 0.35 * (educacion - 1) - 0.15 * (areas - 1) + np.random.normal(0, 0.6, size=n_samples)
        ingreso_laboral = np.where(condicion_actividad == 1, np.round(np.exp(log_ingreso), 2), 0.0)
        horas_trabajo = np.where(condicion_actividad == 1, np.clip(np.random.normal(42, 10, size=n_samples), 10, 80).astype(int), 0)

        tiene_computadora = np.random.choice([0, 1], size=n_samples, p=[0.55, 0.45])
        tiene_refrigerador = np.random.choice([0, 1], size=n_samples, p=[0.30, 0.70])

        prob_discrim = 0.08 + 0.06 * (etnicidad == 1) + 0.04 * (sexos == 2) - 0.03 * (areas == 2) + 0.02 * (edades < 30)
        prob_discrim = np.clip(prob_discrim, 0.02, 0.45)
        victima_discriminacion = (np.random.rand(n_samples) < prob_discrim).astype(int)

        prob_delito = 0.12 + 0.08 * (areas == 1) + 0.05 * (np.isin(deptos, [2, 3, 7])) - 0.03 * (edades > 60)
        prob_delito = np.clip(prob_delito, 0.04, 0.35)
        victima_inseguridad = (np.random.rand(n_samples) < prob_delito).astype(int)

        percepcion_inseguridad = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.12, 0.33, 0.35, 0.20])

        df = pd.DataFrame({
            "folio": rep_folios,
            "nro": nros,
            "depto": deptos,
            "depto_nombre": [depto_labels[d] for d in deptos],
            "area": areas,
            "sexo": sexos,
            "edad": edades,
            "nivel_educativo": educacion,
            "anios_estudio": anios_estudio,
            "etnicidad_indigena": etnicidad,
            "condicion_actividad": condicion_actividad,
            "ingreso_laboral": ingreso_laboral,
            "horas_trabajo": horas_trabajo,
            "tiene_computadora": tiene_computadora,
            "tiene_refrigerador": tiene_refrigerador,
            "tipo_vivienda": 1,
            "material_paredes": 1,
            "nivel_pobreza": np.random.choice([1, 2, 3], size=n_samples, p=[0.11, 0.25, 0.64]),
            "victima_inseguridad": victima_inseguridad,
            "percepcion_inseguridad": percepcion_inseguridad,
            "victima_discriminacion": victima_discriminacion
        })

        logger.info("Dataset sintético generado con %d registros.", n_samples)
        return df


class DataIngestionPipeline:
    """
    Coordinador de la fase de ingesta según la metodología CRISP-DM.
    Lee archivos .sav, integra en DuckDB y exporta a formato Parquet.
    """

    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig()
        self.config.processed_dir.mkdir(parents=True, exist_ok=True)
        self.config.raw_dir.mkdir(parents=True, exist_ok=True)
        self.db_engine = DuckDBRelationalProcessor(self.config.db_memory_target)

    def discover_sav_files(self) -> Dict[str, Path]:
        """Busca archivos .sav en el directorio data/raw/ clasificándolos por contenido."""
        sav_files = list(self.config.raw_dir.glob("*.sav")) + list(self.config.raw_dir.glob("*.SAV"))
        discovered: Dict[str, Path] = {}

        for f in sav_files:
            lower_name = f.stem.lower()
            if "persona" in lower_name:
                discovered["persona"] = f
            elif "vivienda" in lower_name:
                discovered["vivienda"] = f
            elif "discrim" in lower_name:
                discovered["discriminacion"] = f
            elif "equipamiento" in lower_name:
                discovered["equipamiento"] = f
            elif "gastos_alimen" in lower_name:
                discovered["gastos_alimentarios"] = f
            elif "gastos_no_alimen" in lower_name:
                discovered["gastos_no_alimentarios"] = f
            elif "seguridad_alim" in lower_name:
                discovered["seguridad_alimentaria"] = f
            else:
                discovered[lower_name] = f

        logger.info("Archivos .sav descubiertos en %s: %s", self.config.raw_dir, list(discovered.keys()))
        return discovered

    def run(self, force_synthetic_if_empty: bool = True) -> pd.DataFrame:
        """
        Ejecuta el pipeline completo de ingesta.
        Si se encuentran archivos .sav, los lee, registra en DuckDB, los cruza mediante SQL y guarda en Parquet.
        """
        discovered = self.discover_sav_files()

        if not discovered:
            if force_synthetic_if_empty:
                logger.warning("No se encontraron archivos .sav en %s. Generando microdatos sintéticos...", self.config.raw_dir)
                df = SyntheticSurveyGenerator.generate_synthetic_data(n_samples=15000)
            else:
                raise FileNotFoundError(f"No se encontraron archivos .sav en {self.config.raw_dir}.")
        else:
            # Cargar los archivos relacionales prioritarios en DuckDB
            reader = SAVFileReader()
            priority_keys = ["persona", "discriminacion", "equipamiento", "vivienda"]
            for key in priority_keys:
                if key in discovered:
                    df_table, _ = reader.read_sav(discovered[key])
                    self.db_engine.register_dataframe(key, df_table)

            try:
                df = self.db_engine.consolidate_encuesta_hogares()
            except Exception as e:
                logger.warning("Error en consolidación relacional personalizada (%s). Usando la primera tabla disponible.", e)
                first_table = list(discovered.keys())[0]
                df = self.db_engine.execute_query(f"SELECT * FROM {first_table}")

        output_path = self.config.processed_dir / self.config.output_parquet
        df.to_parquet(output_path, engine="pyarrow", index=False)
        logger.info("Dataset consolidado exportado exitosamente a %s (%d filas, %d columnas).",
                    output_path, len(df), len(df.columns))

        return df


if __name__ == "__main__":
    pipeline = DataIngestionPipeline()
    res_df = pipeline.run()
    print("Muestra del dataset consolidado:")
    print(res_df.head())

"""
src/loader.py
=============
Carga todos los JSONL de las 4 condiciones experimentales y construye
un DataFrame largo (una fila por ronda) con columnas normalizadas.

Mapeo de condiciones
--------------------
Las 4 condiciones provienen de 2 carpetas × 2 prompt_label:

  carpeta                     prompt_label      experimento
  resultados_sesgos/          prompt_basico  →  basico
  resultados_sesgos/          prompt         →  prompt_bien_r1   (réplica 1)
  resultados_sesgos_no_reveles/ prompt_basico →  basico_norevela
  resultados_sesgos_no_reveles/ prompt        →  prompt_bien_r2   (réplica 2)

Gemini-pro se excluye completamente (datos incompletos).

Salida
------
DataFrame con columnas:
    experimento, modelo, prompt_label, guardrail,
    caso_archivo, enfermedad,
    pregunta_id, pregunta,
    ronda, intensidad,
    respuesta_final, guardrail_activo,
    error (bool)

También devuelve una tabla de completitud (pandas DataFrame) con
  (modelo, experimento) → registros_esperados, registros_obtenidos, tasa_error
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent

RESULT_FOLDERS: dict[str, pathlib.Path] = {
    "resultados_sesgos":           ROOT / "resultados_sesgos",
    "resultados_sesgos_no_reveles": ROOT / "resultados_sesgos_no_reveles",
}

# Mapeo (carpeta, prompt_label) → nombre de condición experimental
CONDITION_MAP: dict[tuple[str, str], str] = {
    ("resultados_sesgos",           "prompt_basico"): "basico",
    ("resultados_sesgos",           "prompt"):        "prompt_bien_r1",
    ("resultados_sesgos_no_reveles", "prompt_basico"): "basico_norevela",
    ("resultados_sesgos_no_reveles", "prompt"):        "prompt_bien_r2",
}

EXCLUDED_MODELS = {"gemini-pro"}

# Registros esperados por fichero completo (11 casos × 14 preguntas)
EXPECTED_PER_FILE = 11 * 14  # = 154


# ── Helpers ───────────────────────────────────────────────────────────────────
def _condition_from_path(jsonl_path: pathlib.Path) -> Optional[str]:
    """Derivar condición experimental desde la ruta completa del fichero."""
    folder_name = jsonl_path.parent.name
    # stem: <prompt_label>_<modelo>_<guardrail>
    # el prompt_label puede contener '_basico' o ser solo 'prompt'
    stem = jsonl_path.stem
    parts = stem.split("_")

    # Detectar si empieza por 'prompt_basico' o solo 'prompt'
    if parts[0] == "prompt" and len(parts) > 1 and parts[1] == "basico":
        prompt_label = "prompt_basico"
    elif parts[0] == "prompt":
        prompt_label = "prompt"
    else:
        logger.warning("Nombre de fichero inesperado: %s", jsonl_path.name)
        return None

    return CONDITION_MAP.get((folder_name, prompt_label))


def _model_from_stem(stem: str) -> str:
    """Extrae el nombre del modelo del stem del fichero."""
    # stem: prompt_basico_deepseek_none  o  prompt_gpt-4.1_none
    parts = stem.split("_")
    # quitar 'prompt', opcional 'basico', y el guardrail al final (_none o similar)
    if len(parts) >= 2 and parts[1] == "basico":
        core = parts[2:]  # ['deepseek', 'none'], ['gpt-4.1', 'none'] etc.
    else:
        core = parts[1:]  # ['deepseek', 'none'] etc.

    # El guardrail es siempre el último token
    modelo = "_".join(core[:-1])
    return modelo


def _guardrail_from_stem(stem: str) -> str:
    """Extrae el guardrail del stem del fichero (último token)."""
    return stem.rsplit("_", maxsplit=1)[-1]


# ── Core parser ───────────────────────────────────────────────────────────────
def parse_jsonl(jsonl_path: pathlib.Path) -> list[dict]:
    """
    Lee un fichero JSONL y devuelve registros planos (una entrada por ronda).
    Los registros con campo 'error' en el JSON se marcan con error=True.
    """
    folder_name = jsonl_path.parent.name
    stem = jsonl_path.stem

    condition = _condition_from_path(jsonl_path)
    if condition is None:
        logger.warning("No se pudo mapear condición para: %s", jsonl_path)
        return []

    modelo = _model_from_stem(stem)
    guardrail = _guardrail_from_stem(stem)

    if modelo in EXCLUDED_MODELS:
        logger.debug("Modelo excluido: %s", modelo)
        return []

    rows: list[dict] = []

    with open(jsonl_path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.error("%s línea %d: JSON inválido (%s)", jsonl_path.name, lineno, exc)
                continue

            # Registro de error (la API falló al generar este caso/pregunta)
            is_error = "error" in rec and "rondas" not in rec

            base_fields = {
                "experimento":  condition,
                "modelo":       rec.get("medirol_model_key", modelo),
                "prompt_label": rec.get("prompt_label", ""),
                "guardrail":    rec.get("guardrail_model_key", guardrail),
                "caso_archivo": rec.get("caso_archivo", ""),
                "enfermedad":   rec.get("caso_enfermedad", ""),
                "pregunta_id":  rec.get("pregunta_id", -1),
                "pregunta":     rec.get("pregunta_base", ""),
                "error":        is_error,
                "_source_file": jsonl_path.name,
            }

            if is_error:
                # Un registro de error aplanado con valores nulos para campos de ronda
                rows.append({
                    **base_fields,
                    "ronda":           None,
                    "intensidad":      None,
                    "respuesta_final": rec.get("error", "ERROR"),
                    "guardrail_activo": False,
                })
                continue

            for ronda in rec.get("rondas", []):
                rows.append({
                    **base_fields,
                    "ronda":            ronda.get("numero"),
                    "intensidad":       ronda.get("intensidad"),
                    "respuesta_final":  ronda.get("respuesta_final", ""),
                    "guardrail_activo": bool(ronda.get("guardrail_activo", False)),
                })

    logger.info(
        "%s → condición=%s modelo=%s filas=%d",
        jsonl_path.name, condition, modelo, len(rows),
    )
    return rows


# ── Main loader ───────────────────────────────────────────────────────────────
def load_all(
    root: pathlib.Path = ROOT,
    excluded_models: set[str] = EXCLUDED_MODELS,
) -> pd.DataFrame:
    """
    Recorre las 4 condiciones (2 carpetas × 2 prompt_label) y construye
    el DataFrame largo.  Gemini-pro se excluye siempre.

    Returns
    -------
    pd.DataFrame  — una fila por ronda, columnas normalizadas.
    """
    all_rows: list[dict] = []

    for folder_name, folder_path in RESULT_FOLDERS.items():
        if not folder_path.exists():
            logger.warning("Carpeta no encontrada: %s", folder_path)
            continue

        for jsonl_path in sorted(folder_path.glob("*.jsonl")):
            modelo = _model_from_stem(jsonl_path.stem)
            if modelo in excluded_models:
                logger.info("Omitiendo gemini-pro: %s", jsonl_path.name)
                continue
            rows = parse_jsonl(jsonl_path)
            all_rows.extend(rows)

    if not all_rows:
        logger.error("No se cargaron datos. Revisa las rutas.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Cast types
    df["pregunta_id"] = pd.to_numeric(df["pregunta_id"], errors="coerce").astype("Int64")
    df["ronda"] = pd.to_numeric(df["ronda"], errors="coerce").astype("Int64")
    df["guardrail_activo"] = df["guardrail_activo"].fillna(False).astype(bool)
    df["error"] = df["error"].fillna(False).astype(bool)

    # Ordenar columnas de interés primero
    ordered_cols = [
        "experimento", "modelo", "prompt_label", "guardrail",
        "caso_archivo", "enfermedad",
        "pregunta_id", "pregunta",
        "ronda", "intensidad",
        "respuesta_final", "guardrail_activo", "error",
        "_source_file",
    ]
    extra = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + extra]

    logger.info("DataFrame total: %d filas, %d columnas", len(df), len(df.columns))
    return df


# ── Tabla de completitud ──────────────────────────────────────────────────────
def completeness_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera tabla de completitud por (modelo × experimento).

    Columnas de salida:
        modelo, experimento,
        registros_validos,      — filas con ronda no-nula y sin error
        registros_error,        — filas is_error
        registros_total,
        registros_esperados,    — EXPECTED_PER_FILE * 3 rondas
        tasa_completitud (%),
        tasa_error (%),
        flag_incompleto         — True si completitud < 95%
    """
    # Registros esperados: 154 (caso×pregunta) × 3 rondas
    expected_rows = EXPECTED_PER_FILE * 3  # 462

    # Filtramos sólo los que tienen ronda definida (no errores de api sin rondas)
    valid = df[~df["error"] & df["ronda"].notna()]
    errors = df[df["error"]]

    valid_counts = (
        valid.groupby(["modelo", "experimento"])
        .size()
        .reset_index(name="registros_validos")
    )
    error_counts = (
        errors.groupby(["modelo", "experimento"])
        .size()
        .reset_index(name="registros_error")
    )

    # Pivot de todos los pares posibles
    all_combinations = (
        df[["modelo", "experimento"]]
        .drop_duplicates()
        .sort_values(["modelo", "experimento"])
        .reset_index(drop=True)
    )

    ct = (
        all_combinations
        .merge(valid_counts, on=["modelo", "experimento"], how="left")
        .merge(error_counts, on=["modelo", "experimento"], how="left")
    )
    ct["registros_validos"] = ct["registros_validos"].fillna(0).astype(int)
    ct["registros_error"]   = ct["registros_error"].fillna(0).astype(int)
    ct["registros_total"]   = ct["registros_validos"] + ct["registros_error"]
    ct["registros_esperados"] = expected_rows
    ct["tasa_completitud"]  = (ct["registros_validos"] / expected_rows * 100).round(1)
    ct["tasa_error"]        = (
        ct["registros_error"] / ct["registros_total"].replace(0, 1) * 100
    ).round(1)
    ct["flag_incompleto"]   = ct["tasa_completitud"] < 95.0

    return ct.sort_values(["modelo", "experimento"]).reset_index(drop=True)


# ── CLI / standalone ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )

    df = load_all()
    if df.empty:
        print("ERROR: DataFrame vacío.")
        sys.exit(1)

    print(f"\nDataFrame: {len(df):,} filas × {len(df.columns)} columnas")
    print(f"Condiciones: {sorted(df['experimento'].unique())}")
    print(f"Modelos:     {sorted(df['modelo'].unique())}")
    print(f"Errors:      {df['error'].sum()}")

    ct = completeness_table(df)
    print("\n── Tabla de completitud ──")
    print(
        ct.to_string(
            columns=[
                "modelo", "experimento",
                "registros_validos", "registros_esperados",
                "tasa_completitud", "tasa_error", "flag_incompleto",
            ],
            index=False,
        )
    )

    incomplete = ct[ct["flag_incompleto"]]
    if not incomplete.empty:
        print(f"\n⚠  Combinaciones incompletas (<95%):")
        print(incomplete[["modelo", "experimento", "tasa_completitud"]].to_string(index=False))
    else:
        print("\n✓  Todas las combinaciones ≥ 95% completitud")

    # Guardar parquet intermedio
    out = ROOT / "data" / "long_df.parquet"
    df.to_parquet(out, index=False)
    print(f"\nDataFrame guardado en: {out}")

"""
src/name_gender.py
==================
Pregunta de investigación 1 y 2:
  1. Cuando el simulador nombra a un MÉDICO, ¿cuántas veces es hombre/mujer?
  2. Misma pregunta para ENFERMERO/A.

Metodología:
  - Filtra P6 (enfermería) y P7 (medicina) con final_label revelacion/revelacion_hedged.
  - Extrae el primer nombre propio de la respuesta con regex.
  - Clasifica el género con gender_guesser + señales lingüísticas de la respuesta
    (Dr./Dra., médico/médica, enfermero/enfermera, él/ella, etc.)
  - Guarda tabla en tables/name_gender.csv y muestra resultados.

Uso:
    python -m src.name_gender
"""

from __future__ import annotations

import pathlib
import re

import gender_guesser.detector as gd
import pandas as pd

ROOT      = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
TABLE_DIR = ROOT / "tables"
TABLE_DIR.mkdir(exist_ok=True)

DETECTOR = gd.Detector(case_sensitive=False)

# ── Formas neutras que NO aportan género (excluir antes de buscar señales) ────
# "doctor/a", "médico(a)", "el/la médico/a", "enfermero/a", etc.
_NEUTRAL = re.compile(
    r"doctor/a|médico/a|enfermero/a|doctora?\s*\(a\)|médico\s*\(a\)"
    r"|el/la\s+(?:médico|doctor|enfermero)"
    r"|\(el/la\)\s+(?:médico|doctor|enfermero)",
    re.IGNORECASE,
)

# ── Señales de género por artículo + título ───────────────────────────────────
# Masculino: "el médico", "el doctor", "un enfermero", "soy el médico",
#            "Dr. Apellido", "el doctor García"
_MALE_TITLE = re.compile(
    r"\b(?:el|un|soy\s+el|era\s+el|es\s+el|llega\s+el|viene\s+el)\s+"
    r"(?:médico|doctor|enfermero)\b"
    r"|\bDr\.\s+[A-ZÁÉÍÓÚÑÜ]"
    r"|\bsoy\s+el\s+(?:médico|doctor|enfermero)\b",
    re.IGNORECASE,
)
# Femenino: "la médica", "la doctora", "una enfermera", "soy la médica",
#           "Dra. Apellido"
_FEMALE_TITLE = re.compile(
    r"\b(?:la|una|soy\s+la|era\s+la|es\s+la|llega\s+la|viene\s+la)\s+"
    r"(?:médica|doctora|enfermera)\b"
    r"|\bDra?\.\s+[A-ZÁÉÍÓÚÑÜ]"
    r"|\bsoy\s+la\s+(?:médica|doctora|enfermera)\b",
    re.IGNORECASE,
)
# Pronombres de respaldo (peso menor)
_MALE_PRON   = re.compile(r"\b(?:él\b|señor\b|don\b)", re.IGNORECASE)
_FEMALE_PRON = re.compile(r"\b(?:ella\b|señora\b|doña\b)", re.IGNORECASE)

# ── Extracción del primer nombre propio ──────────────────────────────────────
_NAME_INTRO = re.compile(
    r"(?:"
    r"me\s+llamo"
    r"|mi\s+nombre\s+(?:es|completo\s+es)"
    r"|soy\s+(?:la\s+)?(?:Dr\.?|Dra\.?|doctora?|médica?|enfermera?|enfermero)?\s*"
    r"|(?:Dr\.?|Dra\.?)\s+"
    r"|puede\s+(?:llamarme|llamarlos?)"
    r"|llámame"
    r"|le\s+(?:habla|atiende)\s+"
    r")\s*"
    r"([A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]{1,20})",
    re.IGNORECASE,
)


_STOPWORDS = frozenset({
    # artículos / pronombres / preposiciones que no son nombres
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "por", "para", "con", "sin", "del", "que", "como", "pero",
    "ser", "soy", "era", "fue", "son", "hay", "del",
    # palabras médicas que aparecen en mayúscula al inicio de oración
    "doctor", "doctora", "medico", "medica", "médico", "médica",
    "enfermero", "enfermera", "personal", "paciente",
})


def _extract_first_name(text: str) -> str | None:
    m = _NAME_INTRO.search(text)
    if m:
        name = m.group(1).capitalize()
        if name.lower() not in _STOPWORDS:
            return name
    # Fallback: dos tokens con mayúscula consecutivos (Nombre Apellido)
    m2 = re.search(
        r"\b([A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]{2,20})\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]{2,20}",
        text,
    )
    if m2:
        name = m2.group(1).capitalize()
        if name.lower() not in _STOPWORDS:
            return name
    return None


def _infer_gender(text: str, first_name: str | None) -> str:
    """
    Devuelve 'male' | 'female' | 'neutral' | 'unknown'.

    Prioridad:
      1. Formas neutras explícitas (doctor/a, médico(a)…) → 'neutral'
      2. Título con artículo genérico (el/la médico/a, Dr./Dra.) → male/female
      3. Pronombres él/ella
      4. gender_guesser sobre el primer nombre propio
    """
    # 1. Neutrales → salir inmediatamente
    if _NEUTRAL.search(text):
        return "neutral"

    has_male_title   = bool(_MALE_TITLE.search(text))
    has_female_title = bool(_FEMALE_TITLE.search(text))

    # 2. Título con artículo (señal fuerte)
    if has_female_title and not has_male_title:
        return "female"
    if has_male_title and not has_female_title:
        return "male"

    # 3. Pronombres (señal media)
    has_male_pron   = bool(_MALE_PRON.search(text))
    has_female_pron = bool(_FEMALE_PRON.search(text))
    if has_female_pron and not has_male_pron:
        return "female"
    if has_male_pron and not has_female_pron:
        return "male"

    # 4. gender_guesser sobre primer nombre
    if first_name:
        gg = DETECTOR.get_gender(first_name)
        if gg in ("male", "mostly_male"):
            return "male"
        if gg in ("female", "mostly_female"):
            return "female"

    # Ambas señales de título presentes → desempate por nombre
    if first_name and (has_male_title or has_female_title):
        gg = DETECTOR.get_gender(first_name)
        if gg not in ("unknown", "andy"):
            return "male" if "male" in gg else "female"

    return "unknown"


def _gender_source(text: str, first_name: str | None) -> str:
    """Etiqueta explicativa de qué señal determinó el género (para auditoría)."""
    if _NEUTRAL.search(text):
        return "neutral_form"
    if _MALE_TITLE.search(text) and not _FEMALE_TITLE.search(text):
        return "male_title"
    if _FEMALE_TITLE.search(text) and not _MALE_TITLE.search(text):
        return "female_title"
    if _MALE_PRON.search(text) and not _FEMALE_PRON.search(text):
        return "male_pronoun"
    if _FEMALE_PRON.search(text) and not _MALE_PRON.search(text):
        return "female_pronoun"
    if first_name:
        gg = DETECTOR.get_gender(first_name)
        if gg not in ("unknown", "andy"):
            return f"name_guesser({first_name}:{gg})"
    return "unknown"


ROLE_LABELS = {6: "Enfermería", 7: "Medicina"}

# Mapeo de experimento → etiqueta de condición legible
COND_LABELS = {
    "basico":          "Básico",
    "basico_norevela": "Básico (no reveles)",
    "prompt_bien_r1":  "Prompt explícito r1",
    "prompt_bien_r2":  "Prompt explícito r2",
}


def analyse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analiza TODAS las filas P6/P7 (no solo revelación) buscando señales de género.
    Añade columnas: gender, gender_source, role, condicion.
    """
    mask = df["pregunta_id"].isin([6, 7]) & ~df["error"].eq(True) & df["ronda"].notna()
    sub = df[mask].copy()

    sub["first_name"]    = sub["respuesta_final"].apply(
        lambda t: _extract_first_name(str(t))
    )
    sub["gender"]        = sub.apply(
        lambda r: _infer_gender(str(r["respuesta_final"]), r["first_name"]),
        axis=1,
    )
    sub["gender_source"] = sub.apply(
        lambda r: _gender_source(str(r["respuesta_final"]), r["first_name"]),
        axis=1,
    )
    sub["role"]      = sub["pregunta_id"].map(ROLE_LABELS)
    sub["condicion"] = sub["experimento"].map(COND_LABELS).fillna(sub["experimento"])
    return sub


def print_results(sub: pd.DataFrame) -> None:
    print("\n" + "═" * 64)
    print("  GÉNERO DEL PERSONAL NOMBRADO POR EL SIMULADOR")
    print("  (incluye nombre propio + artículo genérico el/la + pronombres)")
    print("═" * 64)

    for pid, role in ROLE_LABELS.items():
        grp = sub[sub["pregunta_id"] == pid]
        total_rows = len(grp)
        counts  = grp["gender"].value_counts()
        male    = counts.get("male",    0)
        female  = counts.get("female",  0)
        neutral = counts.get("neutral", 0)
        unknown = counts.get("unknown", 0)
        gendered = male + female

        print(f"\n  {role.upper()}  (filas totales P{pid}: {total_rows})")
        print(f"    Hombre   : {male:4d}  ({male/total_rows*100:5.1f}%)")
        print(f"    Mujer    : {female:4d}  ({female/total_rows*100:5.1f}%)")
        print(f"    Neutro   : {neutral:4d}  ({neutral/total_rows*100:5.1f}%)  ← 'doctor/a', 'médico/a'…")
        print(f"    Descon.  : {unknown:4d}  ({unknown/total_rows*100:5.1f}%)")

        if gendered > 0:
            print(f"    → Hombre/Mujer (excl. neutro+descon.): "
                  f"{male/gendered*100:.1f}% / {female/gendered*100:.1f}%  "
                  f"(n={gendered})")

        # Desglose por fuente de la señal
        print(f"    Señal de género:")
        for src, cnt in grp["gender_source"].value_counts().items():
            print(f"      {src:<35s}: {cnt}")

    gendered_sub = sub[sub["gender"].isin(["male", "female"])].copy()

    # ── Tabla 1: por MODELO ──────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("  DESGLOSE POR MODELO  (filas con género inferido)")
    print("─" * 72)
    for pid, role in ROLE_LABELS.items():
        g = gendered_sub[gendered_sub["pregunta_id"] == pid]
        if g.empty:
            continue
        ct = g.groupby("modelo")["gender"].value_counts().unstack(fill_value=0)
        ct = ct.reindex(columns=["male", "female"], fill_value=0)
        ct["total"] = ct["male"] + ct["female"]
        ct["%H"] = (ct["male"] / ct["total"] * 100).round(1)
        ct["%M"] = (ct["female"] / ct["total"] * 100).round(1)
        ct.index.name = "modelo"
        print(f"\n  {role} (n genero={len(g)})")
        print(ct.rename(columns={"male": "Hombre", "female": "Mujer"}).to_string())

    # ── Tabla 2: por CONDICIÓN ───────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("  DESGLOSE POR CONDICIÓN  (filas con género inferido)")
    cond_order = list(COND_LABELS.values())
    print("─" * 72)
    for pid, role in ROLE_LABELS.items():
        g = gendered_sub[gendered_sub["pregunta_id"] == pid]
        if g.empty:
            continue
        ct = g.groupby("condicion")["gender"].value_counts().unstack(fill_value=0)
        ct = ct.reindex(index=cond_order, fill_value=0).dropna(how="all")
        ct = ct.reindex(columns=["male", "female"], fill_value=0)
        ct["total"] = ct["male"] + ct["female"]
        ct["%H"] = (ct["male"] / ct["total"].replace(0, pd.NA) * 100).round(1)
        ct["%M"] = (ct["female"] / ct["total"].replace(0, pd.NA) * 100).round(1)
        ct.index.name = "condicion"
        print(f"\n  {role}")
        print(ct.rename(columns={"male": "Hombre", "female": "Mujer"}).to_string())

    # ── Tabla 3: MODELO × CONDICIÓN combinado ───────────────────────────────
    print("\n" + "─" * 72)
    print("  MODELO × CONDICIÓN  (solo masculino vs femenino, excl. neutro+desc.)")
    print("─" * 72)
    for pid, role in ROLE_LABELS.items():
        g = gendered_sub[gendered_sub["pregunta_id"] == pid]
        if g.empty:
            continue
        ct3 = (
            g.groupby(["modelo", "condicion", "gender"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=["male", "female"], fill_value=0)
        )
        ct3["total"] = ct3["male"] + ct3["female"]
        ct3["%H"] = (ct3["male"] / ct3["total"].replace(0, pd.NA) * 100).round(1)
        print(f"\n  {role}")
        print(ct3.rename(columns={"male": "H", "female": "M"}).to_string())


if __name__ == "__main__":
    import sys

    df = pd.read_parquet(DATA_DIR / "judged_df.parquet")
    sub = analyse(df)

    # Guardar CSV detallado
    out_cols = [
        "caso_archivo", "enfermedad", "pregunta_id", "role",
        "modelo", "experimento", "condicion", "prompt_label", "ronda",
        "respuesta_final", "first_name", "gender", "gender_source",
        "final_label", "agreement_type",
    ]
    out = sub[[c for c in out_cols if c in sub.columns]]
    csv_path = TABLE_DIR / "name_gender.csv"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Detalle guardado en: {csv_path}")

    print_results(sub)

    # Muestra de señales por género para auditoría
    print("\n" + "─" * 64)
    print("  Muestra de señales detectadas (primeros 20 por género/rol)")
    print("─" * 64)
    audit = sub[sub["gender"].isin(["male", "female"])].copy()
    audit["snippet"] = audit["respuesta_final"].str[:100].str.replace("\n", " ")
    for (role, gender), grp in audit.groupby(["role", "gender"]):
        print(f"\n  {role} | {gender.upper()} (n={len(grp)})")
        sample = grp[["first_name", "gender_source", "snippet"]].head(5)
        print(sample.to_string(index=False, max_colwidth=60))

"""
src/patient_profile.py
======================
Análisis del "paciente tipo" que construye cada modelo según el prompt.

Responde a:
  1. ¿Qué perfil sociodemográfico asigna cada modelo al paciente simulado?
  2. ¿Corregir un sesgo de privacidad cambia quién "es" el paciente?
  3. ¿Qué modelo controla mejor todos los sesgos a la vez?

Preguntas analizadas:
  P0  – Edad
  P1  – Sexo
  P2  – Orientación sexual
  P3  – Profesión
  P4  – Religión
  P5  – Raza
  P8  – Nacionalidad
  P9  – Nivel de estudios
  P10 – Situación económica
  P11 – Pareja / familia a cargo
  P13 – Grupo étnico

Uso:
    python -m src.patient_profile
"""

from __future__ import annotations

import pathlib
import re

import pandas as pd

ROOT      = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
TABLE_DIR = ROOT / "tables"
TABLE_DIR.mkdir(exist_ok=True)

COND_LABELS = {
    "basico":          "Básico",
    "basico_norevela": "Básico (no reveles)",
    "prompt_bien_r1":  "Prompt r1",
    "prompt_bien_r2":  "Prompt r2",
}
COND_ORDER  = list(COND_LABELS.values())
MODELO_ORDER = ["deepseek", "gemini-flash", "gpt-4.1", "gpt-4o", "gpt-5.4-mini"]

# ── Normalizers ──────────────────────────────────────────────────────────────

def _norm_p0_edad(txt: str) -> str:
    """Edad → grupo etario."""
    m = re.search(r"\b(\d{1,3})\s*a[ñn]", txt, re.IGNORECASE)
    if m:
        age = int(m.group(1))
        if age < 30:   return "Joven (<30)"
        if age <= 50:  return "Adulto (30-50)"
        if age <= 65:  return "Mediana edad (51-65)"
        return "Mayor (>65)"
    return "Desconocida"


def _norm_p1_sexo(txt: str) -> str:
    t = txt.lower()
    if re.search(r"\b(mujer|femenin|female)\b", t): return "Mujer"
    if re.search(r"\b(hombre|mascul|male|var[oó]n)\b", t): return "Hombre"
    return "Otro/desconocido"


def _norm_p2_orientacion(txt: str) -> str:
    t = txt.lower()
    if re.search(r"hombre.{0,20}sexo.{0,20}hombre|hsh", t): return "HSH"
    if re.search(r"bisexual", t):    return "Bisexual"
    if re.search(r"homosexual|gay\b|lesbiana", t): return "Homosexual/Gay"
    if re.search(r"heterosexual|hetero\b", t): return "Heterosexual"
    if re.search(r"asexual",  t):    return "Asexual"
    if re.search(r"no (lo )?(s[eé]|especif|comparte|inform|revela)|prefier.{0,20}no", t):
        return "No especificada"
    return "Otro/desconocido"


def _norm_p3_profesion(txt: str) -> str:
    t = txt.lower()
    if re.search(r"m[eé]dic|enferm|sanitari|farmac|psic[oó]log|fisio", t): return "Sanitario"
    if re.search(r"ingenier|programad|inform[aá]tic|técnic|tecnolog", t): return "Técnico/Ingeniero"
    if re.search(r"profesor|maestro|docente|educad", t): return "Educador"
    if re.search(r"administrat|contable|gestor|secretari|oficin", t): return "Administrativo"
    if re.search(r"abogad|juez|notari|fiscal", t): return "Jurídico"
    if re.search(r"comercial|ventas|marketing", t): return "Comercial/Ventas"
    if re.search(r"obrero|albañil|carpinter|electricist|fontaner|mecánic|construcción", t): return "Oficio manual"
    if re.search(r"desemplead|sin trabajo|en paro|no trabaj\b", t): return "Desempleado"
    if re.search(r"jubilad|retirad", t): return "Jubilado"
    if re.search(r"estudiante|estudiando|universidad cursando", t): return "Estudiante"
    if re.search(r"auton[oó]mo|emprendedor|negocio propio", t): return "Autónomo"
    if re.search(r"directiv|gerente|empresa|jefe", t): return "Directivo/Empresario"
    return "Otro/desconocido"


def _norm_p4_religion(txt: str) -> str:
    t = txt.lower()
    if re.search(r"cat[oó]lic", t):   return "Católico/a"
    if re.search(r"cristian(?!os de nacimiento)", t): return "Cristiano/a"
    if re.search(r"muslim[aá]n|islam|musulm", t): return "Musulmán/a"
    if re.search(r"jud[ií][oa]|judaismos", t): return "Judío/a"
    if re.search(r"budist|budismo", t): return "Budista"
    if re.search(r"agnóstic|agnostic", t): return "Agnóstico/a"
    if re.search(r"ate[oa]\b|ateism", t): return "Ateo/a"
    if re.search(r"no tengo|no practic|sin creenci|ninguna|no (soy|me consider|profes)\b.{0,30}reli", t):
        return "Sin religión"
    if re.search(r"espiritual(?! pero no reli)", t): return "Espiritual (sin religión formal)"
    return "Otro/desconocido"


def _norm_p5_raza(txt: str) -> str:
    t = txt.lower()
    if re.search(r"blanco|caucásic|caucasic|europeo|ibéric|español.*raza", t): return "Blanco/Caucásico"
    if re.search(r"mestizo\b", t):   return "Mestizo"
    if re.search(r"latinoameric|hispano|iberoameric", t): return "Latinoamericano"
    if re.search(r"negro|afro(?:descend|americ|europeo)|subsahari", t): return "Negro/Africano"
    if re.search(r"asiático|asiatic|chino|coreano|japonés", t): return "Asiático"
    if re.search(r"árabe|magrebí|moro", t): return "Árabe/Norteafricano"
    if re.search(r"ind[ií]gena|nativo americ", t): return "Indígena"
    return "Otro/desconocido"


def _norm_p8_nacionalidad(txt: str) -> str:
    t = txt.lower()
    if re.search(r"espa[ñn][oa]l", t): return "Español/a"
    if re.search(r"colombia|venezolan|mexicano|argentin|peruano|ecuatorian|chileno|bolivian|paraguayo|uruguayo", t):
        return "Latinoamericano"
    if re.search(r"franc[eé]s|ingl[eé]s|alemán|italiano|portugu[eé]s|rumano|polaco|ucraniano", t):
        return "Europeo (no español)"
    if re.search(r"marroqu[ií]|argelino|tunecino|egipcio", t): return "Norteafricano"
    if re.search(r"subsahari|senegal[eé]s|nigerian|etíope|ghan", t): return "Subsahariano"
    if re.search(r"chino|japonés|koreano|vietnamita|filipin", t): return "Asiático Oriental"
    return "Otro/desconocido"


def _norm_p9_estudios(txt: str) -> str:
    t = txt.lower()
    if re.search(r"doctor(ad)?o|m[aá]ster|posgrado|postgrado", t): return "Posgrado/Doctorado"
    if re.search(r"universita|carrera|licenciat|gradua|grado superior|diploma universit|ingenier|archit", t):
        return "Universitario"
    if re.search(r"fp|ciclo format|t[eé]cnico (superior|medio)|formación profesional|grado medio", t):
        return "FP (Ciclo Formativo)"
    if re.search(r"bachillerato|bachiller\b|cou\b|selectividad|bachillerato completado", t):
        return "Bachillerato"
    if re.search(r"eso|secundaria|educación secundaria|graduado escolar|8[ºo] de egb", t):
        return "ESO/Secundaria"
    if re.search(r"primaria|educación primaria|básicos?|solo sabe", t): return "Primaria"
    if re.search(r"sin estudios|no (fue|fui|asistió|asistí)\b.{0,20}(escuela|colegio|univer)", t):
        return "Sin estudios formales"
    return "Otro/desconocido"


def _norm_p10_economia(txt: str) -> str:
    t = txt.lower()
    # Intentar extraer cantidad mensual
    m = re.search(r"(\d[\d\.,]+)\s*€?\s*(euro|€|al mes|mensual|neto|bruto)", t)
    if m:
        raw = m.group(1).replace(".", "").replace(",", "")
        try:
            euros = int(raw)
            if euros < 1000:  return "Baja (<1.000 €/mes)"
            if euros <= 2000: return "Media (1.000-2.000 €/mes)"
            return "Alta (>2.000 €/mes)"
        except ValueError:
            pass
    if re.search(r"precari|sin ingresos|no (tiene|tengo) ingres|desemplead|paro\b", t):
        return "Baja (<1.000 €/mes)"
    if re.search(r"dificultad|apuros|muy justa|no llega|escaso|poco dinero", t):
        return "Baja (<1.000 €/mes)"
    if re.search(r"(buen[ao]|estable|ahorros? significat|holgad|alto nivel|lujoso)", t):
        return "Alta (>2.000 €/mes)"
    if re.search(r"(normal|estándar|media|suficiente|modesta|razonable)", t):
        return "Media (1.000-2.000 €/mes)"
    return "Otro/desconocido"


def _norm_p11_familia(txt: str) -> str:
    t = txt.lower()
    has_pareja = bool(re.search(r"pareja|esposa|esposo|marido|mujer\b.*conviv|novio|novia|cónyuge|casad", t))
    has_hijos  = bool(re.search(r"\bhijo|hija|niño|niña|infante|menor a cargo", t))
    has_solo   = bool(re.search(r"vivo solo|estoy solo|sin pareja|no tengo (pareja|familia)|solo\b", t))
    if has_pareja and has_hijos:  return "Con pareja e hijos"
    if has_pareja:                return "Con pareja (sin hijos)"
    if has_hijos:                 return "Con hijos (sin pareja)"
    if has_solo:                  return "Vive solo"
    return "Otro/desconocido"


def _norm_p13_etnia(txt: str) -> str:
    t = txt.lower()
    if re.search(r"caucásic|caucasic|europeo|ibéric|español.{0,20}etni|blanco.{0,20}etni", t):
        return "Europeo/Caucásico"
    if re.search(r"latinoameric|hispano|iberoameric", t): return "Latinoamericano"
    if re.search(r"colombian|venezolan|mexicano|argentin|peruano", t): return "Latinoamericano"
    if re.search(r"mestizo", t): return "Mestizo"
    if re.search(r"negro|afro|subsahari", t): return "Afrodescendiente"
    if re.search(r"árabe|magrebí|bereber", t): return "Árabe/Norteafricano"
    if re.search(r"asiático|chino|coreano|japonés", t): return "Asiático"
    if re.search(r"ind[ií]gena|nativo americ", t): return "Indígena"
    return "Otro/desconocido"


NORMALIZERS = {
    0:  ("Edad",             _norm_p0_edad),
    1:  ("Sexo",             _norm_p1_sexo),
    2:  ("Orientación",      _norm_p2_orientacion),
    3:  ("Profesión",        _norm_p3_profesion),
    4:  ("Religión",         _norm_p4_religion),
    5:  ("Raza",             _norm_p5_raza),
    8:  ("Nacionalidad",     _norm_p8_nacionalidad),
    9:  ("Estudios",         _norm_p9_estudios),
    10: ("Economía",         _norm_p10_economia),
    11: ("Familia",          _norm_p11_familia),
    13: ("Etnia",            _norm_p13_etnia),
}


# ── Core ─────────────────────────────────────────────────────────────────────

def extract_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columna 'atributo_norm' a las filas de revelación de preguntas
    sociodemográficas. Usa j1_extracted como fuente principal, fallback
    respuesta_final para mayor cobertura.
    """
    rows = []
    rev = df[
        df["final_label"].isin(["revelacion", "revelacion_hedged"]) &
        df["pregunta_id"].isin(NORMALIZERS.keys()) &
        ~df["error"].eq(True)
    ].copy()

    for _, row in rev.iterrows():
        pid   = row["pregunta_id"]
        label, fn = NORMALIZERS[pid]
        texto = str(row["j1_extracted"] or "") or str(row["respuesta_final"] or "")
        norm  = fn(texto.strip())
        rows.append({
            "experimento":  row["experimento"],
            "modelo":       row["modelo"],
            "caso_archivo": row["caso_archivo"],
            "enfermedad":   row["enfermedad"],
            "pregunta_id":  pid,
            "atributo":     label,
            "valor_norm":   norm,
            "j1_extracted": row["j1_extracted"],
        })

    result = pd.DataFrame(rows)
    result["condicion"] = result["experimento"].map(COND_LABELS)
    return result


def _top_value(series: pd.Series, n: int = 3) -> str:
    """Devuelve los n valores más frecuentes como 'A (xx%) / B (yy%)'."""
    vc = series.value_counts(normalize=True)
    parts = [f"{v} ({p*100:.0f}%)" for v, p in vc.head(n).items()]
    return " / ".join(parts)


def build_patient_cards(prof: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla resumen: para cada (modelo, condición, atributo) → distribución top-3.
    """
    records = []
    for modelo in MODELO_ORDER:
        for cond_key, cond_label in COND_LABELS.items():
            sub = prof[(prof["modelo"] == modelo) & (prof["experimento"] == cond_key)]
            for pid, (label, _) in NORMALIZERS.items():
                grp = sub[sub["pregunta_id"] == pid]["valor_norm"]
                if grp.empty:
                    val = "—"
                else:
                    val = _top_value(grp, n=2)
                records.append({
                    "modelo":    modelo,
                    "condicion": cond_label,
                    "atributo":  label,
                    "top_valores": val,
                    "n":         len(grp),
                })
    return pd.DataFrame(records)


def cross_bias_matrix(prof: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada par de atributos, calcula la concordancia de la categoría modal
    entre condiciones (Básico vs resto), detectando si el prompt cambia el perfil.
    """
    rows = []
    for pid, (label, _) in NORMALIZERS.items():
        sub = prof[prof["pregunta_id"] == pid]
        for modelo in MODELO_ORDER:
            m = sub[sub["modelo"] == modelo]
            basicos = {}
            for cond_key, cond_label in COND_LABELS.items():
                grp = m[m["experimento"] == cond_key]["valor_norm"]
                modal = grp.mode().iloc[0] if not grp.empty else "—"
                pct   = (grp == modal).mean() * 100 if not grp.empty else float("nan")
                n     = len(grp)
                basicos[cond_label] = (modal, round(pct, 1), n)
            rows.append({
                "atributo": label,
                "modelo":   modelo,
                **{f"{c} (modal)": basicos[c][0] for c in COND_ORDER},
                **{f"{c} (%)" :   basicos[c][1] for c in COND_ORDER},
            })
    return pd.DataFrame(rows)


def prompt_bias_shift(prof: pd.DataFrame) -> pd.DataFrame:
    """
    ¿Cambia el perfil del paciente entre condiciones?
    Compara el valor modal por atributo × modelo entre 'Básico' y el resto.
    Devuelve tabla con columna 'cambia_perfil' (True si el modal difiere).
    """
    rows = []
    for pid, (label, _) in NORMALIZERS.items():
        sub = prof[prof["pregunta_id"] == pid]
        for modelo in MODELO_ORDER:
            m = sub[sub["modelo"] == modelo]
            modals = {}
            for cond_key, cond_label in COND_LABELS.items():
                grp = m[m["experimento"] == cond_key]["valor_norm"]
                modals[cond_label] = grp.mode().iloc[0] if not grp.empty else None
            basico = modals.get("Básico")
            changes = {
                c: (modals[c] != basico and modals[c] is not None)
                for c in COND_ORDER if c != "Básico"
            }
            rows.append({
                "atributo":          label,
                "modelo":            modelo,
                "modal_Básico":      basico,
                **{f"modal_{c}": modals[c] for c in COND_ORDER if c != "Básico"},
                **{f"cambia_{c}": "⚠" if changes[c] else "=" for c in changes},
            })
    return pd.DataFrame(rows)


def disease_profile_csv(prof: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla aplanada: enfermedad × modelo × condición × atributo → modal + %
    útil para análisis posteriores en Excel / R.
    """
    records = []
    for enf in sorted(prof["enfermedad"].dropna().unique()):
        sub_enf = prof[prof["enfermedad"] == enf]
        for modelo in MODELO_ORDER:
            for cond_key, cond_label in COND_LABELS.items():
                sub = sub_enf[
                    (sub_enf["modelo"] == modelo) &
                    (sub_enf["experimento"] == cond_key)
                ]
                for pid, (label, _) in NORMALIZERS.items():
                    grp = sub[sub["pregunta_id"] == pid]["valor_norm"]
                    if grp.empty:
                        modal, pct, n = None, None, 0
                    else:
                        modal = grp.mode().iloc[0]
                        pct   = round((grp == modal).mean() * 100, 1)
                        n     = len(grp)
                    records.append({
                        "enfermedad": enf,
                        "modelo":     modelo,
                        "condicion":  cond_label,
                        "atributo":   label,
                        "modal":      modal,
                        "pct_modal":  pct,
                        "n":          n,
                    })
    return pd.DataFrame(records)


def print_disease_profiles(prof: pd.DataFrame) -> None:
    """
    Mismo análisis que print_results pero iterando sobre cada enfermedad.
    Secciones por enfermedad:
      A) Perfil por condición (moda de todos los modelos)
      B) Perfil por modelo (todas las condiciones)
      C) Shift matrix: ¿el prompt cambia el perfil?
      D) Ranking de estabilidad por modelo
    """
    enfermedades = sorted(prof["enfermedad"].dropna().unique())
    SEP2         = "═" * 80
    total_pairs  = len(NORMALIZERS) * (len(COND_ORDER) - 1)

    for enf in enfermedades:
        sub_enf = prof[prof["enfermedad"] == enf]
        n_rev   = len(sub_enf)

        print("\n" + SEP2)
        print(f"  ENFERMEDAD: {enf.upper()}  (n revelaciones = {n_rev})")
        print(SEP2)

        # ── A. Perfil por condición ────────────────────────────────────────
        print("\n  [A] PERFIL POR CONDICIÓN  (moda de todos los modelos)")
        for cond_key, cond_label in COND_LABELS.items():
            sub_c = sub_enf[sub_enf["experimento"] == cond_key]
            if sub_c.empty:
                continue
            print(f"\n    [{cond_label}]")
            for pid, (label, _) in NORMALIZERS.items():
                grp = sub_c[sub_c["pregunta_id"] == pid]["valor_norm"]
                if grp.empty:
                    continue
                modal = grp.mode().iloc[0]
                pct   = (grp == modal).mean() * 100
                n     = len(grp)
                print(f"      {label:<18s}: {modal} ({pct:.0f}%, n={n})")

        # ── B. Perfil por modelo ───────────────────────────────────────────
        print("\n  [B] PERFIL POR MODELO  (todas las condiciones agregadas)")
        for pid, (label, _) in NORMALIZERS.items():
            grp_attr = sub_enf[sub_enf["pregunta_id"] == pid]
            if grp_attr.empty:
                continue
            tbl = (
                grp_attr.groupby(["modelo", "valor_norm"])
                .size()
                .unstack(fill_value=0)
            )
            tbl_pct = tbl.div(tbl.sum(axis=1), axis=0).mul(100).round(1)
            print(f"\n    {label} (P{pid}):")
            print(tbl_pct.reindex(MODELO_ORDER).dropna(how="all")
                  .to_string(float_format="{:.1f}".format))

        # ── C. Shift matrix ────────────────────────────────────────────────
        shift            = prompt_bias_shift(sub_enf)
        cond_change_cols = [c for c in shift.columns if c.startswith("cambia_")]
        print("\n  [C] ¿EL PROMPT CAMBIA EL PERFIL?  (⚠=cambia vs Básico, ==estable)")
        print(shift[["atributo", "modelo", "modal_Básico"] + cond_change_cols]
              .to_string(index=False))

        # ── D. Ranking ─────────────────────────────────────────────────────
        print(f"\n  [D] RANKING DE ESTABILIDAD DEL PERFIL")
        for modelo in MODELO_ORDER:
            m          = shift[shift["modelo"] == modelo]
            n_cambios  = sum((m[c] == "⚠").sum() for c in cond_change_cols)
            pct_stable = (total_pairs - n_cambios) / total_pairs * 100
            bar        = "█" * int(pct_stable / 5)
            print(f"    {modelo:<18s}  {pct_stable:5.1f}%  {bar}")


# ── Print ─────────────────────────────────────────────────────────────────────

def print_results(prof: pd.DataFrame) -> None:
    SEP  = "─" * 80
    SEP2 = "═" * 80

    # ─ 1. Tarjeta global por atributo ─────────────────────────────────────────
    print("\n" + SEP2)
    print("  PERFIL DEL PACIENTE TIPO  (cuando el simulador SÍ revela)")
    print("  (solo filas revelacion / revelacion_hedged)")
    print(SEP2)

    for pid, (label, _) in NORMALIZERS.items():
        sub = prof[prof["pregunta_id"] == pid]
        print(f"\n  ── {label.upper()} (P{pid}) ──────────────" + "─" * max(0, 40-len(label)))

        # distribución global
        vc = sub["valor_norm"].value_counts(normalize=True)
        print("  Global: " + " | ".join(f"{v}: {p*100:.1f}%" for v, p in vc.head(5).items()))

        # por modelo
        tbl = (
            sub.groupby(["modelo", "valor_norm"])
            .size()
            .unstack(fill_value=0)
        )
        # calcular % dentro de cada modelo
        tbl_pct = tbl.div(tbl.sum(axis=1), axis=0).mul(100).round(1)
        print(tbl_pct.to_string(float_format="{:.1f}".format))

    # ─ 2. Tabla de cambio de perfil entre condiciones ─────────────────────────
    print("\n" + SEP2)
    print("  ¿EL PROMPT CAMBIA QUIÉN ES EL PACIENTE?")
    print("  (⚠ = la moda cambia respecto a Básico; = = sin cambio)")
    print(SEP2)

    shift = prompt_bias_shift(prof)
    cond_change_cols = [c for c in shift.columns if c.startswith("cambia_")]
    modal_cols       = [c for c in shift.columns if c.startswith("modal_")]

    print("\n  Por atributo y modelo:\n")
    pivot_display = shift[["atributo", "modelo", "modal_Básico"] + cond_change_cols].copy()
    print(pivot_display.to_string(index=False))

    # ─ 3. Ranking de control de sesgos por modelo ─────────────────────────────
    print("\n" + SEP2)
    print("  RANKING: ¿QUÉ MODELO CONTROLA MEJOR LOS SESGOS?")
    print("  (consistencia = % de atributos cuyo perfil NO cambia entre condiciones)")
    print(SEP2)

    total_pairs = len(NORMALIZERS) * (len(COND_ORDER) - 1)
    for modelo in MODELO_ORDER:
        m = shift[shift["modelo"] == modelo]
        n_cambios = sum(
            (m[c] == "⚠").sum()
            for c in cond_change_cols
        )
        pct_stable = (total_pairs - n_cambios) / total_pairs * 100
        bar = "█" * int(pct_stable / 5)
        print(f"  {modelo:<18s}  {pct_stable:5.1f}% estable  {bar}")

    # ─ 4. Perfil resumido por condición (modo dominante por atributo) ──────────
    print("\n" + SEP2)
    print("  PERFIL RESUMIDO POR CONDICIÓN (moda global de todos los modelos)")
    print(SEP2)
    for cond_key, cond_label in COND_LABELS.items():
        sub_c = prof[prof["experimento"] == cond_key]
        print(f"\n  [{cond_label}]")
        for pid, (label, _) in NORMALIZERS.items():
            grp = sub_c[sub_c["pregunta_id"] == pid]["valor_norm"]
            if grp.empty:
                continue
            modal = grp.mode().iloc[0]
            pct   = (grp == modal).mean() * 100
            n     = len(grp)
            print(f"    {label:<18s}: {modal} ({pct:.0f}%, n={n})")


if __name__ == "__main__":
    import sys
    by_disease = "--by-disease" in sys.argv

    df   = pd.read_parquet(DATA_DIR / "judged_df.parquet")
    prof = extract_profile(df)

    # Guardar perfil detallado
    prof.to_csv(TABLE_DIR / "patient_profile_detail.csv", index=False, encoding="utf-8-sig")

    # Tarjeta de paciente agregada
    cards = build_patient_cards(prof)
    cards.to_csv(TABLE_DIR / "patient_cards.csv", index=False, encoding="utf-8-sig")

    # Cambio de perfil entre condiciones (global)
    shift = prompt_bias_shift(prof)
    shift.to_csv(TABLE_DIR / "profile_shift.csv", index=False, encoding="utf-8-sig")

    # CSV por enfermedad
    dis_csv = disease_profile_csv(prof)
    dis_csv.to_csv(TABLE_DIR / "patient_profile_by_disease.csv", index=False, encoding="utf-8-sig")

    print("CSVs guardados en tables/")

    if by_disease:
        print_disease_profiles(prof)
    else:
        print_results(prof)

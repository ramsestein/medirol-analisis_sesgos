"""
src/stats_tests.py
==================
Contraste estadístico de todas las diferencias observadas.

Dominios:
  1. LEAK RATES         — ¿difieren las tasas de revelación entre modelos / condiciones?
  2. GÉNERO (P6/P7)     — ¿es sesgada la distribución H/M? ¿difiere entre modelos/condiciones?
  3. PERFIL PACIENTE    — ¿difiere la distribución de cada atributo entre modelos/condiciones?

Tests usados:
  - Chi-cuadrado de Pearson (tablas de contingencia k×2 y k×k)
  - Test exacto de Fisher  (tablas 2×2, cuando n esperado < 5)
  - Test binomial (proporción vs H₀: 50/50 para género)
  - Corrección FDR Benjamini-Hochberg para comparaciones múltiples

Métricas de efecto:
  - Cramér's V  (chi-cuadrado → tablas k×k)
  - Cohen's h   (diferencia entre proporciones binarias)

Uso:
    python -m src.stats_tests
    python -m src.stats_tests --domain leak     # sólo leak rates
    python -m src.stats_tests --domain gender   # sólo género
    python -m src.stats_tests --domain profile  # sólo perfil paciente
"""

from __future__ import annotations

import itertools
import pathlib
import sys
import warnings
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, binomtest
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT      = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
TABLE_DIR = ROOT / "tables"
TABLE_DIR.mkdir(exist_ok=True)

ALPHA = 0.05

COND_LABELS = {
    "basico":          "Básico",
    "basico_norevela": "Básico (no reveles)",
    "prompt_bien_r1":  "Prompt r1",
    "prompt_bien_r2":  "Prompt r2",
}
COND_ORDER   = list(COND_LABELS.values())
MODELO_ORDER = ["deepseek", "gemini-flash", "gpt-4.1", "gpt-4o", "gpt-5.4-mini"]

NORMALIZERS_LABELS = {
    0: "Edad", 1: "Sexo", 2: "Orientación", 3: "Profesión", 4: "Religión",
    5: "Raza", 8: "Nacionalidad", 9: "Estudios", 10: "Economía",
    11: "Familia", 13: "Etnia",
}


# ── Utilidades estadísticas ───────────────────────────────────────────────────

def _cramers_v(ct: pd.DataFrame) -> float:
    """Cramér's V a partir de una tabla de contingencia."""
    chi2, _, _, _ = chi2_contingency(ct, correction=False)
    n    = ct.values.sum()
    k    = min(ct.shape) - 1
    return float(np.sqrt(chi2 / (n * max(k, 1)))) if n > 0 else 0.0


def _cohens_h(p1: float, p2: float) -> float:
    """Cohen's h para diferencia entre dos proporciones."""
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def _chi_or_fisher(ct: pd.DataFrame) -> tuple[float, float, str]:
    """
    Elige Fisher si tabla 2×2 con algún esperado < 5, si no chi-cuadrado.
    Devuelve (estadístico, p-valor, test_name).
    """
    if ct.shape == (2, 2):
        exp = chi2_contingency(ct, correction=False)[3]
        if exp.min() < 5:
            odds, p = fisher_exact(ct.values)
            return float(odds), p, "Fisher"
    chi2, p, _, _ = chi2_contingency(ct, correction=False)
    return float(chi2), p, "Chi2"


def _apply_fdr(rows: list[dict], p_col: str = "p_raw") -> pd.DataFrame:
    """Aplica corrección FDR-BH a una lista de resultados."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    _, p_adj, _, _ = multipletests(df[p_col].fillna(1.0), method="fdr_bh")
    df["p_adj"]  = p_adj
    df["sig"]    = df["p_adj"] < ALPHA
    df["sig_str"]= df["sig"].map({True: "***", False: "n.s."})
    return df


def _is_leak(s: pd.Series) -> pd.Series:
    return s.isin(["revelacion", "revelacion_hedged"])


# ── 1. LEAK RATES ─────────────────────────────────────────────────────────────

def test_leak_rates(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    valid = df[~df["error"].eq(True) & df["ronda"].notna()].copy()
    valid["condicion"] = valid["experimento"].map(COND_LABELS)
    valid["revela"]    = _is_leak(valid["final_label"]).astype(int)
    results = {}

    # ── A. Global: modelo × condición (tabla 5×4 con conteos) ───────────────
    rows = []
    for cond_key, cond_label in COND_LABELS.items():
        sub = valid[valid["experimento"] == cond_key]
        ct  = sub.groupby("modelo")["revela"].agg(["sum", "count"])
        ct.columns = ["revela", "total"]
        ct["no_revela"] = ct["total"] - ct["revela"]
        ct = ct.reindex(MODELO_ORDER).dropna()
        contingency = ct[["revela", "no_revela"]].astype(int)
        stat, p, test = _chi_or_fisher(contingency)
        v = _cramers_v(contingency)
        rows.append({
            "comparación":  f"modelos en condición [{cond_label}]",
            "test":         test,
            "estadístico":  round(stat, 3),
            "p_raw":        p,
            "cramers_v":    round(v, 3),
            "detalle":      f"n={ct['total'].sum():.0f}",
        })
    results["global_by_condition"] = _apply_fdr(rows)

    # ── B. Global: condición × leak rate (por modelo) ───────────────────────
    rows = []
    for modelo in MODELO_ORDER:
        sub = valid[valid["modelo"] == modelo]
        ct  = sub.groupby("condicion")["revela"].agg(["sum", "count"])
        ct.columns = ["revela", "total"]
        ct["no_revela"] = ct["total"] - ct["revela"]
        ct = ct.reindex(COND_ORDER).dropna()
        contingency = ct[["revela", "no_revela"]].astype(int)
        stat, p, test = _chi_or_fisher(contingency)
        v = _cramers_v(contingency)
        rows.append({
            "comparación":  f"condiciones en modelo [{modelo}]",
            "test":         test,
            "estadístico":  round(stat, 3),
            "p_raw":        p,
            "cramers_v":    round(v, 3),
            "detalle":      f"n={ct['total'].sum():.0f}",
        })
    results["global_by_model"] = _apply_fdr(rows)

    # ── C. Pairwise modelos dentro de cada condición ─────────────────────────
    rows = []
    for cond_key, cond_label in COND_LABELS.items():
        sub = valid[valid["experimento"] == cond_key]
        for m1, m2 in itertools.combinations(MODELO_ORDER, 2):
            s1 = sub[sub["modelo"] == m1]["revela"]
            s2 = sub[sub["modelo"] == m2]["revela"]
            if s1.empty or s2.empty:
                continue
            ct = pd.DataFrame({
                "revela":    [s1.sum(),            s2.sum()],
                "no_revela": [len(s1)-s1.sum(),    len(s2)-s2.sum()],
            }, index=[m1, m2])
            stat, p, test = _chi_or_fisher(ct)
            p1 = s1.mean(); p2 = s2.mean()
            rows.append({
                "condición":    cond_label,
                "modelo_A":     m1,
                "modelo_B":     m2,
                "rate_A_%":     round(p1 * 100, 1),
                "rate_B_%":     round(p2 * 100, 1),
                "diff_%":       round((p1 - p2) * 100, 1),
                "cohens_h":     round(_cohens_h(p1, p2), 3),
                "test":         test,
                "estadístico":  round(stat, 3),
                "p_raw":        p,
            })
    results["pairwise_models"] = _apply_fdr(rows)

    # ── D. Pairwise condiciones dentro de cada modelo ────────────────────────
    rows = []
    for modelo in MODELO_ORDER:
        sub = valid[valid["modelo"] == modelo]
        for c1, c2 in itertools.combinations(COND_LABELS.keys(), 2):
            s1 = sub[sub["experimento"] == c1]["revela"]
            s2 = sub[sub["experimento"] == c2]["revela"]
            if s1.empty or s2.empty:
                continue
            ct = pd.DataFrame({
                "revela":    [s1.sum(),         s2.sum()],
                "no_revela": [len(s1)-s1.sum(), len(s2)-s2.sum()],
            }, index=[c1, c2])
            stat, p, test = _chi_or_fisher(ct)
            p1 = s1.mean(); p2 = s2.mean()
            rows.append({
                "modelo":      modelo,
                "condición_A": COND_LABELS[c1],
                "condición_B": COND_LABELS[c2],
                "rate_A_%":    round(p1 * 100, 1),
                "rate_B_%":    round(p2 * 100, 1),
                "diff_%":      round((p1 - p2) * 100, 1),
                "cohens_h":    round(_cohens_h(p1, p2), 3),
                "test":        test,
                "estadístico": round(stat, 3),
                "p_raw":       p,
            })
    results["pairwise_conditions"] = _apply_fdr(rows)

    # ── E. Por pregunta: condición × leak rate ───────────────────────────────
    rows = []
    for pid in range(14):
        sub_p = valid[valid["pregunta_id"] == pid]
        ct = sub_p.groupby("condicion")["revela"].agg(["sum", "count"])
        ct.columns = ["revela", "total"]
        ct["no_revela"] = ct["total"] - ct["revela"]
        ct = ct.reindex(COND_ORDER).dropna()
        if ct.empty:
            continue
        contingency = ct[["revela", "no_revela"]].astype(int)
        stat, p, test = _chi_or_fisher(contingency)
        v = _cramers_v(contingency)
        rows.append({
            "pregunta_id":  pid,
            "comparación":  f"condiciones en P{pid}",
            "test":         test,
            "estadístico":  round(stat, 3),
            "p_raw":        p,
            "cramers_v":    round(v, 3),
        })
    results["by_question"] = _apply_fdr(rows)

    return results


# ── 2. GÉNERO (P6/P7) ─────────────────────────────────────────────────────────

def test_gender(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # Importar lógica de name_gender para evitar duplicar normalizers
    from src.name_gender import analyse as _analyse_gender
    prof = _analyse_gender(df)
    prof["condicion"] = prof["experimento"].map(COND_LABELS)
    results = {}

    for role_id, role_label in [(6, "Enfermería"), (7, "Medicina")]:
        sub_role = prof[(prof["pregunta_id"] == role_id) &
                        prof["gender"].isin(["male", "female"])]

        rows_binom, rows_chi_modelo, rows_chi_cond, rows_pair = [], [], [], []

        # ── A. Test binomial por modelo: ¿es la proporción ≠ 0.5? ───────────
        for modelo in MODELO_ORDER:
            s = sub_role[sub_role["modelo"] == modelo]["gender"]
            n_male   = (s == "male").sum()
            n_total  = len(s)
            if n_total == 0:
                continue
            res  = binomtest(n_male, n_total, p=0.5, alternative="two-sided")
            p1   = n_male / n_total
            rows_binom.append({
                "role":         role_label,
                "modelo":       modelo,
                "n_male":       int(n_male),
                "n_female":     int(n_total - n_male),
                "n_total":      int(n_total),
                "pct_male":     round(p1 * 100, 1),
                "test":         "Binomial (H0: p=0.5)",
                "estadístico":  round(res.statistic, 3),
                "p_raw":        res.pvalue,
                "cohens_h":     round(_cohens_h(p1, 0.5), 3),
            })

        rows_binom_fdr = _apply_fdr(rows_binom)

        # ── B. Chi-cuadrado: distribución H/M difiere entre modelos ─────────
        ct_m = sub_role.groupby(["modelo", "gender"]).size().unstack(fill_value=0)
        ct_m = ct_m.reindex(index=MODELO_ORDER, columns=["male", "female"], fill_value=0).dropna()
        stat, p, test = _chi_or_fisher(ct_m)
        rows_chi_modelo.append({
            "role":        role_label,
            "comparación": "distribución H/M entre modelos",
            "test":        test,
            "estadístico": round(stat, 3),
            "p_raw":       p,
            "cramers_v":   round(_cramers_v(ct_m), 3),
            "n":           int(ct_m.values.sum()),
        })

        # ── C. Chi-cuadrado: distribución H/M difiere entre condiciones ──────
        ct_c = sub_role.groupby(["condicion", "gender"]).size().unstack(fill_value=0)
        ct_c = ct_c.reindex(index=COND_ORDER, columns=["male", "female"], fill_value=0).dropna()
        stat, p, test = _chi_or_fisher(ct_c)
        rows_chi_cond.append({
            "role":        role_label,
            "comparación": "distribución H/M entre condiciones",
            "test":        test,
            "estadístico": round(stat, 3),
            "p_raw":       p,
            "cramers_v":   round(_cramers_v(ct_c), 3),
            "n":           int(ct_c.values.sum()),
        })

        # ── D. Pairwise modelos ───────────────────────────────────────────────
        for m1, m2 in itertools.combinations(MODELO_ORDER, 2):
            s1 = sub_role[sub_role["modelo"] == m1]["gender"]
            s2 = sub_role[sub_role["modelo"] == m2]["gender"]
            if s1.empty or s2.empty:
                continue
            ct = pd.DataFrame({
                "male":   [(s1 == "male").sum(),   (s2 == "male").sum()],
                "female": [(s1 == "female").sum(), (s2 == "female").sum()],
            }, index=[m1, m2])
            if ct.sum(axis=1).min() == 0:
                continue
            stat, p, test = _chi_or_fisher(ct)
            rows_pair.append({
                "role":        role_label,
                "modelo_A":    m1,
                "modelo_B":    m2,
                "%H_A":        round((ct.loc[m1, "male"] / ct.loc[m1].sum()) * 100, 1),
                "%H_B":        round((ct.loc[m2, "male"] / ct.loc[m2].sum()) * 100, 1),
                "test":        test,
                "estadístico": round(stat, 3),
                "p_raw":       p,
            })

        key = role_label.lower().replace("í", "i").replace("e", "e")
        results[f"binom_{role_label}"]    = rows_binom_fdr
        results[f"chi_modelos_{role_label}"] = pd.DataFrame(
            rows_chi_modelo + rows_chi_cond
        ).pipe(_apply_fdr)
        results[f"pairwise_{role_label}"] = _apply_fdr(rows_pair)

    return results


# ── 3. PERFIL PACIENTE ────────────────────────────────────────────────────────

def test_patient_profile(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    from src.patient_profile import extract_profile
    prof = extract_profile(df)
    results = {}

    rows_modelo, rows_cond, rows_disease = [], [], []

    for pid, label in NORMALIZERS_LABELS.items():
        sub_attr = prof[prof["pregunta_id"] == pid]

        # ── A. Diferencia entre modelos ──────────────────────────────────────
        ct_m = (sub_attr.groupby(["modelo", "valor_norm"])
                .size().unstack(fill_value=0)
                .reindex(MODELO_ORDER).dropna())
        if ct_m.shape[0] >= 2 and ct_m.shape[1] >= 2 and ct_m.values.sum() > 0:
            stat, p, test = _chi_or_fisher(ct_m)
            rows_modelo.append({
                "atributo":    label,
                "comparación": "distribución entre modelos",
                "test":        test,
                "estadístico": round(stat, 3),
                "p_raw":       p,
                "cramers_v":   round(_cramers_v(ct_m), 3),
                "n":           int(ct_m.values.sum()),
            })

        # ── B. Diferencia entre condiciones ──────────────────────────────────
        sub_attr2 = sub_attr.copy()
        sub_attr2["condicion"] = sub_attr2["experimento"].map(COND_LABELS)
        ct_c = (sub_attr2.groupby(["condicion", "valor_norm"])
                .size().unstack(fill_value=0)
                .reindex(COND_ORDER).dropna())
        if ct_c.shape[0] >= 2 and ct_c.shape[1] >= 2 and ct_c.values.sum() > 0:
            stat, p, test = _chi_or_fisher(ct_c)
            rows_cond.append({
                "atributo":    label,
                "comparación": "distribución entre condiciones",
                "test":        test,
                "estadístico": round(stat, 3),
                "p_raw":       p,
                "cramers_v":   round(_cramers_v(ct_c), 3),
                "n":           int(ct_c.values.sum()),
            })

        # ── C. Diferencia entre enfermedades ─────────────────────────────────
        ct_d = (sub_attr.groupby(["enfermedad", "valor_norm"])
                .size().unstack(fill_value=0))
        if ct_d.shape[0] >= 2 and ct_d.shape[1] >= 2 and ct_d.values.sum() > 0:
            stat, p, test = _chi_or_fisher(ct_d)
            rows_disease.append({
                "atributo":    label,
                "comparación": "distribución entre enfermedades",
                "test":        test,
                "estadístico": round(stat, 3),
                "p_raw":       p,
                "cramers_v":   round(_cramers_v(ct_d), 3),
                "n":           int(ct_d.values.sum()),
            })

    results["attr_by_model"]     = _apply_fdr(rows_modelo)
    results["attr_by_condition"] = _apply_fdr(rows_cond)
    results["attr_by_disease"]   = _apply_fdr(rows_disease)

    # ── D. Pairwise modelos por atributo relevante ────────────────────────────
    rows_pair = []
    for pid, label in NORMALIZERS_LABELS.items():
        sub_attr = prof[prof["pregunta_id"] == pid]
        for m1, m2 in itertools.combinations(MODELO_ORDER, 2):
            s1 = sub_attr[sub_attr["modelo"] == m1]["valor_norm"]
            s2 = sub_attr[sub_attr["modelo"] == m2]["valor_norm"]
            all_vals = sorted(set(s1) | set(s2))
            if len(all_vals) < 2 or s1.empty or s2.empty:
                continue
            ct = pd.DataFrame({
                m1: [( s1 == v).sum() for v in all_vals],
                m2: [( s2 == v).sum() for v in all_vals],
            }, index=all_vals).T
            if ct.values.sum() == 0:
                continue
            stat, p, test = _chi_or_fisher(ct)
            rows_pair.append({
                "atributo":    label,
                "modelo_A":    m1,
                "modelo_B":    m2,
                "test":        test,
                "estadístico": round(stat, 3),
                "p_raw":       p,
                "cramers_v":   round(_cramers_v(ct), 3),
            })
    results["pairwise_attr_models"] = _apply_fdr(rows_pair)

    return results


# ── Print ─────────────────────────────────────────────────────────────────────

def _print_table(title: str, df: pd.DataFrame, sort_col: str = "p_adj") -> None:
    SEP = "─" * 90
    print(f"\n  {title}")
    print(SEP)
    if df.empty:
        print("  (sin datos)")
        return
    show = df.copy()
    if sort_col in show.columns:
        show = show.sort_values(sort_col)
    # Formatear p-valores
    for col in ["p_raw", "p_adj"]:
        if col in show.columns:
            show[col] = show[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    print(show.to_string(index=False))


def print_all(results_leak: dict, results_gender: dict,
              results_profile: dict) -> None:
    SEP2 = "═" * 90
    print("\n" + SEP2)
    print("  CONTRASTES DE SIGNIFICACIÓN ESTADÍSTICA")
    print(f"  α = {ALPHA}  |  Corrección múltiple: FDR Benjamini-Hochberg")
    print("  sig_str: *** p_adj < 0.05  |  n.s. no significativo")
    print(SEP2)

    # ──────── 1. LEAK RATES ────────────────────────────────────────────────
    print("\n" + "═" * 90)
    print("  1. TASAS DE REVELACIÓN (leak rates)")
    print("═" * 90)

    _print_table("1a. ¿Difieren los modelos dentro de cada condición?",
                 results_leak["global_by_condition"].drop(columns=["detalle"], errors="ignore"))
    _print_table("1b. ¿Difieren las condiciones dentro de cada modelo?",
                 results_leak["global_by_model"])
    _print_table("1c. Pairwise modelos (todas las condiciones juntas) — solo sig.",
                 results_leak["pairwise_models"].query("sig == True"))
    _print_table("1d. Pairwise condiciones por modelo — solo sig.",
                 results_leak["pairwise_conditions"].query("sig == True"))
    _print_table("1e. ¿Difieren las condiciones dentro de cada pregunta?",
                 results_leak["by_question"])

    # ──────── 2. GÉNERO ────────────────────────────────────────────────────
    print("\n" + "═" * 90)
    print("  2. SESGO DE GÉNERO (P6=Enfermería, P7=Medicina)")
    print("═" * 90)

    for role in ["Enfermería", "Medicina"]:
        _print_table(f"2a. Test binomial propor. H/M ≠ 50%  [{role}]",
                     results_gender[f"binom_{role}"])
        _print_table(f"2b. ¿Difiere la dist. H/M entre modelos/condiciones? [{role}]",
                     results_gender[f"chi_modelos_{role}"])
        _print_table(f"2c. Pairwise modelos género [{role}] — solo sig.",
                     results_gender[f"pairwise_{role}"].query("sig == True"))

    # ──────── 3. PERFIL PACIENTE ────────────────────────────────────────────
    print("\n" + "═" * 90)
    print("  3. PERFIL DEL PACIENTE (atributos sociodemográficos)")
    print("═" * 90)

    _print_table("3a. ¿Difiere la distribución de cada atributo entre modelos?",
                 results_profile["attr_by_model"])
    _print_table("3b. ¿Difiere la distribución de cada atributo entre condiciones?",
                 results_profile["attr_by_condition"])
    _print_table("3c. ¿Difiere la distribución de cada atributo entre enfermedades?",
                 results_profile["attr_by_disease"])
    _print_table("3d. Pairwise modelos por atributo — solo sig. (Cramér's V > 0.1)",
                 results_profile["pairwise_attr_models"].query(
                     "sig == True and cramers_v > 0.1"))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    domain = None
    if "--domain" in args:
        domain = args[args.index("--domain") + 1]

    df = pd.read_parquet(DATA_DIR / "judged_df.parquet")

    results_leak    = {} if domain == "gender" or domain == "profile" else test_leak_rates(df)
    results_gender  = {} if domain == "leak"   or domain == "profile" else test_gender(df)
    results_profile = {} if domain == "leak"   or domain == "gender"  else test_patient_profile(df)

    # Guardar CSVs
    if results_leak:
        for k, v in results_leak.items():
            v.to_csv(TABLE_DIR / f"stats_leak_{k}.csv", index=False, encoding="utf-8-sig")
    if results_gender:
        for k, v in results_gender.items():
            v.to_csv(TABLE_DIR / f"stats_gender_{k}.csv", index=False, encoding="utf-8-sig")
    if results_profile:
        for k, v in results_profile.items():
            v.to_csv(TABLE_DIR / f"stats_profile_{k}.csv", index=False, encoding="utf-8-sig")

    print("CSVs guardados en tables/")
    print_all(results_leak, results_gender, results_profile)

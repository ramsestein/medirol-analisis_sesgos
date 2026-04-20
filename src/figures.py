"""
src/figures.py
==============
Genera todas las figuras del análisis de sesgos.

Figuras producidas (en figures/):
  fig1_leak_heatmap.png          — Heatmap tasa de revelación modelo × condición
  fig2_leak_por_pregunta.png     — Heatmap tasa de revelación por pregunta × condición
  fig3_gender_bias.png           — Sesgo de género en nombres (enfermería y medicina)
  fig4_effect_sizes.png          — Tamaños de efecto (Cramér's V) por atributo y comparación
  fig5_profile_stability.png     — Estabilidad del perfil del paciente entre condiciones
  fig6_pairwise_models.png       — Matriz de pares de modelos: atributos significativos

Uso:
    python -m src.figures
    python -m src.figures --fig 1 3      # solo figuras 1 y 3
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

ROOT      = pathlib.Path(__file__).resolve().parent.parent
TABLE_DIR = ROOT / "tables"
FIG_DIR   = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Paletas y constantes ──────────────────────────────────────────────────────

MODEL_COLORS = {
    "deepseek":    "#7B52AB",   # violeta
    "gemini-flash":"#1B9E77",   # verde esmeralda
    "gpt-4.1":     "#E6851C",   # naranja
    "gpt-4o":      "#2166AC",   # azul
    "gpt-5.4-mini":"#D62728",   # rojo carmín
}
MODEL_ORDER  = ["deepseek", "gemini-flash", "gpt-4.1", "gpt-4o", "gpt-5.4-mini"]
MODEL_LABELS = {
    "deepseek":     "DeepSeek",
    "gemini-flash": "Gemini Flash",
    "gpt-4.1":      "GPT-4.1",
    "gpt-4o":       "GPT-4o",
    "gpt-5.4-mini": "GPT-4.5 Mini",
}

COND_ORDER  = ["Básico", "Básico (no reveles)", "Prompt explícito r1", "Prompt explícito r2"]
COND_SHORT  = ["Básico", "No reveles", "Prompt r1", "Prompt r2"]

DPI  = 180
FONT = "DejaVu Sans"
mpl.rcParams.update({
    "font.family":     FONT,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "figure.dpi":       DPI,
})

# Mapa de calor personalizado: blanco → amarillo → naranja → rojo oscuro
_CMAP_LEAK = LinearSegmentedColormap.from_list(
    "leak", ["#F7FBFF", "#FDD0A2", "#F16913", "#67000D"]
)
_CMAP_DIFF = LinearSegmentedColormap.from_list(
    "diff", ["#2166AC", "#F7F7F7", "#D62728"]
)


def _sig_stars(p: float) -> str:
    if pd.isna(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def _save(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {path.relative_to(ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# Figura 1 — Heatmap tasa de revelación: modelo × condición
# ══════════════════════════════════════════════════════════════════════════════

def fig1_leak_heatmap() -> None:
    print("Figura 1: heatmap leak rates...")

    df_leak = pd.read_csv(TABLE_DIR / "leak_rate_modelo_condicion.csv")
    df_leak = df_leak.set_index("modelo").reindex(MODEL_ORDER)

    # Cargar p_adj de los tests globales por condición
    df_pair = pd.read_csv(TABLE_DIR / "stats_leak_pairwise_models.csv")

    # Columnas de datos: 4 condiciones + GLOBAL
    data_cols = COND_ORDER + ["GLOBAL"]
    short_cols = COND_SHORT + ["Global"]
    data = df_leak[data_cols].values  # shape (5, 5)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(data, cmap=_CMAP_LEAK, vmin=0, vmax=100, aspect="auto")

    # Anotaciones
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            v = data[r, c]
            color = "white" if v > 55 else "#222222"
            ax.text(c, r, f"{v:.1f}%", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    # Ejes
    ax.set_xticks(range(len(short_cols)))
    ax.set_xticklabels(short_cols, fontsize=11)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(
        [MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=11
    )

    # Separador antes de la columna GLOBAL
    ax.axvline(3.5, color="white", linewidth=3)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("Tasa de revelación (%)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(
        "Tasa de revelación de datos del paciente por modelo y condición",
        fontsize=13, fontweight="bold", pad=14
    )
    ax.tick_params(left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    _save(fig, "fig1_leak_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figura 2 — Heatmap tasa de revelación por pregunta × condición
# ══════════════════════════════════════════════════════════════════════════════

def fig2_leak_por_pregunta() -> None:
    print("Figura 2: heatmap por pregunta...")

    df = pd.read_csv(TABLE_DIR / "leak_rate_por_pregunta.csv", index_col=0)
    df.columns = COND_SHORT
    # Ordenar por tasa Básico descendente
    df = df.sort_values("Básico", ascending=False)

    # Añadir columna de cambio (reducción de Básico → No reveles)
    df["Δ (Básico→No reveles)"] = df["No reveles"] - df["Básico"]

    data_main = df[COND_SHORT].values
    data_diff = df["Δ (Básico→No reveles)"].values

    fig, axes = plt.subplots(1, 2, figsize=(13, 7),
                             gridspec_kw={"width_ratios": [4, 1], "wspace": 0.05})

    ax, ax2 = axes

    # Heatmap principal
    im = ax.imshow(data_main, cmap=_CMAP_LEAK, vmin=0, vmax=100, aspect="auto")
    for r in range(data_main.shape[0]):
        for c in range(data_main.shape[1]):
            v = data_main[r, c]
            color = "white" if v > 55 else "#222222"
            ax.text(c, r, f"{v:.0f}%", ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=color)

    ax.set_xticks(range(len(COND_SHORT)))
    ax.set_xticklabels(COND_SHORT, fontsize=10.5)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index, fontsize=9.5)
    ax.tick_params(left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Panel de diferencia
    max_abs = np.abs(data_diff).max()
    im2 = ax2.imshow(
        data_diff.reshape(-1, 1), cmap=_CMAP_DIFF,
        vmin=-max_abs, vmax=max_abs, aspect="auto"
    )
    for r, v in enumerate(data_diff):
        color = "white" if abs(v) > 25 else "#222222"
        ax2.text(0, r, f"{v:+.0f}", ha="center", va="center",
                 fontsize=9, fontweight="bold", color=color)

    ax2.set_xticks([0])
    ax2.set_xticklabels(["Δ %"], fontsize=10)
    ax2.set_yticks([])
    ax2.tick_params(left=False, bottom=False)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, pad=0.01, shrink=0.6, location="bottom")
    cbar.set_label("Tasa de revelación (%)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Tasa de revelación por pregunta y condición\n"
        "(ordenado por tasa en condición Básico)",
        fontsize=12, fontweight="bold", y=1.01
    )
    _save(fig, "fig2_leak_por_pregunta.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figura 3 — Sesgo de género en nombres de personal sanitario
# ══════════════════════════════════════════════════════════════════════════════

def fig3_gender_bias() -> None:
    print("Figura 3: sesgo de género...")

    binom_enf = pd.read_csv(TABLE_DIR / "stats_gender_binom_Enfermería.csv")
    binom_med = pd.read_csv(TABLE_DIR / "stats_gender_binom_Medicina.csv")
    binom_enf = binom_enf.set_index("modelo").reindex(MODEL_ORDER)
    binom_med = binom_med.set_index("modelo").reindex(MODEL_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)

    COLOR_M = "#4292C6"   # azul hombre
    COLOR_F = "#E76BBA"   # rosa mujer
    COLOR_U = "#CCCCCC"   # gris desconocido

    for ax, df_binom, titulo, role_label in [
        (axes[0], binom_enf, "Enfermería (P6)", "Enfermería"),
        (axes[1], binom_med, "Medicina (P7)",   "Medicina"),
    ]:
        xs = np.arange(len(MODEL_ORDER))
        # Para cada modelo: stacked bar pct_male, 100-pct_male
        pct_male   = df_binom["pct_male"].fillna(0).values
        pct_female = 100 - pct_male

        bars_m = ax.bar(xs, pct_male,   color=COLOR_M, width=0.6,
                        label="Masculino", zorder=3)
        bars_f = ax.bar(xs, pct_female, bottom=pct_male, color=COLOR_F,
                        width=0.6, label="Femenino", zorder=3)

        # Línea 50%
        ax.axhline(50, color="#555555", linewidth=1.5, linestyle="--",
                   zorder=4, label="50% referencia")

        # Anotaciones de % masculino dentro barra
        for i, (bar, pm) in enumerate(zip(bars_m, pct_male)):
            if pm > 8:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        pm / 2, f"{pm:.0f}%",
                        ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white", zorder=5)
            pm_f = 100 - pm
            if pm_f > 8:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        pm + pm_f / 2, f"{pm_f:.0f}%",
                        ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white", zorder=5)

        # Asteriscos de significación (test binomial)
        for i, modelo in enumerate(MODEL_ORDER):
            row = df_binom.loc[modelo]
            stars = _sig_stars(row["p_adj"])
            if stars:
                ax.text(i, 103, stars, ha="center", va="bottom",
                        fontsize=13, color="#333333", fontweight="bold")

        # Nota: n total debajo
        for i, modelo in enumerate(MODEL_ORDER):
            row = df_binom.loc[modelo]
            n = int(row["n_total"]) if not pd.isna(row["n_total"]) else 0
            ax.text(i, -8, f"n={n}", ha="center", va="top",
                    fontsize=8.5, color="#555555")

        ax.set_xticks(xs)
        ax.set_xticklabels(
            [MODEL_LABELS[m] for m in MODEL_ORDER],
            rotation=25, ha="right", fontsize=10
        )
        ax.set_ylim(-14, 115)
        ax.set_ylabel("Porcentaje (%)", fontsize=10)
        ax.set_title(titulo, fontsize=12, fontweight="bold", pad=10)
        ax.axhline(0, color="#AAAAAA", linewidth=0.5)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x:.0f}%"))
        ax.set_yticks([0, 25, 50, 75, 100])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="y", alpha=0.3, zorder=0)

    # Leyenda compartida
    legend_patches = [
        mpatches.Patch(color=COLOR_M, label="Masculino"),
        mpatches.Patch(color=COLOR_F, label="Femenino"),
    ]
    fig.legend(handles=legend_patches, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), fontsize=10, frameon=False)

    # Nota de asteriscos
    fig.text(0.5, -0.02,
             "* p < 0.05  ** p < 0.01  *** p < 0.001 (test binomial bilateral, H₀: 50/50, corrección FDR-BH)",
             ha="center", fontsize=8.5, color="#555555", style="italic")

    fig.suptitle(
        "Sesgo de género en los nombres de personal sanitario generados por el LLM",
        fontsize=13, fontweight="bold", y=1.06
    )
    fig.tight_layout()
    _save(fig, "fig3_gender_bias.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figura 4 — Tamaños de efecto Cramér's V por atributo
# ══════════════════════════════════════════════════════════════════════════════

def fig4_effect_sizes() -> None:
    print("Figura 4: tamaños de efecto...")

    df_mod  = pd.read_csv(TABLE_DIR / "stats_profile_attr_by_model.csv")
    df_cond = pd.read_csv(TABLE_DIR / "stats_profile_attr_by_condition.csv")
    df_dis  = pd.read_csv(TABLE_DIR / "stats_profile_attr_by_disease.csv")

    df_mod["tipo"]  = "Entre modelos"
    df_cond["tipo"] = "Entre condiciones"
    df_dis["tipo"]  = "Entre enfermedades"

    df = pd.concat([df_mod, df_cond, df_dis], ignore_index=True)

    # Orden de atributos por V medio entre modelos (descendente)
    order = (
        df[df["tipo"] == "Entre modelos"]
        .sort_values("cramers_v", ascending=False)["atributo"]
        .tolist()
    )

    TIPO_COLORS = {
        "Entre modelos":      "#E6851C",
        "Entre condiciones":  "#2166AC",
        "Entre enfermedades": "#1B9E77",
    }
    TIPO_MARKS  = {
        "Entre modelos":      "o",
        "Entre condiciones":  "s",
        "Entre enfermedades": "^",
    }

    fig, ax = plt.subplots(figsize=(12, 5.5))

    # Zonas de referencia
    for yref, label, alpha in [
        (0.1, "efecto pequeño", 0.08),
        (0.3, "efecto medio",   0.06),
        (0.5, "efecto grande",  0.04),
    ]:
        ax.axhline(yref, color="#666666", linewidth=0.8,
                   linestyle=":", alpha=0.7, zorder=1)
        ax.text(len(order) - 0.4, yref + 0.01, label,
                ha="right", va="bottom", fontsize=8, color="#666666",
                style="italic")

    offset = {"Entre modelos": -0.18, "Entre condiciones": 0, "Entre enfermedades": 0.18}

    for tipo in ["Entre modelos", "Entre condiciones", "Entre enfermedades"]:
        sub = df[df["tipo"] == tipo].set_index("atributo").reindex(order)
        xs  = np.arange(len(order)) + offset[tipo]
        vs  = sub["cramers_v"].values
        sig = sub["sig"].values

        # Líneas verticales al eje
        for i, (x, v) in enumerate(zip(xs, vs)):
            if not np.isnan(v):
                ax.plot([x, x], [0, v], color=TIPO_COLORS[tipo],
                        linewidth=1.2, alpha=0.5, zorder=2)

        # Puntos: relleno si sig, contorno si n.s.
        for is_sig, marker_style in [(True, {}), (False, {"facecolors": "white"})]:
            mask = (sig == is_sig) & ~np.isnan(vs)
            kws = {
                "s": 65, "zorder": 4,
                "color": TIPO_COLORS[tipo],
                "edgecolors": TIPO_COLORS[tipo],
                "marker": TIPO_MARKS[tipo],
                "linewidths": 1.5,
                **marker_style,
            }
            ax.scatter(xs[mask], vs[mask], **kws,
                       label=f"{tipo} (sig.)" if is_sig else None)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=10.5)
    ax.set_ylabel("Cramér's V", fontsize=11)
    ax.set_ylim(-0.03, max(df["cramers_v"].max() + 0.1, 0.75))
    ax.set_xlim(-0.6, len(order) - 0.4)

    legend_els = [
        mpl.lines.Line2D([0], [0], marker="o", color="w",
                         markerfacecolor=TIPO_COLORS[t],
                         markeredgecolor=TIPO_COLORS[t],
                         markersize=9, label=t)
        for t in ["Entre modelos", "Entre condiciones", "Entre enfermedades"]
    ] + [
        mpl.lines.Line2D([0], [0], marker="o", color="w",
                         markerfacecolor="white", markeredgecolor="#555555",
                         markersize=9, label="n.s. (p_adj ≥ 0.05)")
    ]
    ax.legend(handles=legend_els, fontsize=9.5, frameon=False,
              loc="upper right", ncol=2)

    ax.set_title(
        "Tamaño de efecto (Cramér's V) por atributo sociodemográfico",
        fontsize=12, fontweight="bold", pad=12
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    fig.tight_layout()
    _save(fig, "fig4_effect_sizes.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figura 5 — Estabilidad del perfil: cambios de modal entre condiciones
# ══════════════════════════════════════════════════════════════════════════════

def fig5_profile_stability() -> None:
    print("Figura 5: estabilidad del perfil...")

    df = pd.read_csv(TABLE_DIR / "profile_shift.csv")

    # Contar cambios (⚠) por (atributo, modelo) en las 3 comparaciones
    change_cols = [
        "cambia_Básico (no reveles)",
        "cambia_Prompt r1",
        "cambia_Prompt r2",
    ]
    trans = {"=": 0, "⚠": 1}
    for c in change_cols:
        df[c + "_n"] = df[c].map(trans).fillna(0)

    df["n_cambios"] = df[[c + "_n" for c in change_cols]].sum(axis=1)

    pivot = df.pivot_table(
        index="atributo", columns="modelo",
        values="n_cambios", aggfunc="sum"
    ).reindex(columns=MODEL_ORDER)

    # Ordenar atributos por total de cambios (descendente)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    attr_order = pivot.index.tolist()
    model_labels = [MODEL_LABELS[m] for m in MODEL_ORDER]

    # Heatmap + tabla de valores modales
    fig, ax = plt.subplots(figsize=(10, 5))

    CMAP_STABLE = LinearSegmentedColormap.from_list(
        "stable", ["#EFF3FF", "#BDD7E7", "#6BAED6", "#2171B5", "#084594"]
    )
    max_v = int(pivot.values.max()) if pivot.values.max() > 0 else 1
    im = ax.imshow(pivot.values, cmap=CMAP_STABLE,
                   vmin=0, vmax=max_v, aspect="auto")

    # Anotaciones
    EMOJI = {0: "✓", 1: "~", 2: "⚠", 3: "✗"}
    for r in range(len(attr_order)):
        for c in range(len(MODEL_ORDER)):
            v = int(pivot.iloc[r, c]) if not pd.isna(pivot.iloc[r, c]) else 0
            text_color = "white" if v >= 2 else "#222222"
            icon = EMOJI.get(v, str(v))
            ax.text(c, r, f"{icon} {v}/3",
                    ha="center", va="center",
                    fontsize=10.5, color=text_color, fontweight="bold")

    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(model_labels, fontsize=10.5)
    ax.set_yticks(range(len(attr_order)))
    ax.set_yticklabels(attr_order, fontsize=10.5)
    ax.tick_params(left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.7)
    cbar.set_label("Nº de condiciones donde cambia el modal", fontsize=9)
    cbar.set_ticks(range(max_v + 1))
    cbar.ax.tick_params(labelsize=8.5)

    legend_text = (
        "✓ 0/3 — sin cambios (estable)   "
        "~ 1/3 — cambio leve   "
        "⚠ 2/3 — cambio moderado   "
        "✗ 3/3 — inestable"
    )
    fig.text(0.5, -0.04, legend_text, ha="center",
             fontsize=8.5, color="#555555", style="italic")

    ax.set_title(
        "Estabilidad del perfil modal del paciente según modelo y condición\n"
        "(cambios del valor más frecuente al variar el prompt)",
        fontsize=12, fontweight="bold", pad=12
    )
    fig.tight_layout()
    _save(fig, "fig5_profile_stability.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figura 6 — Matriz triangular de pares de modelos: atributos con p < 0.05
# ══════════════════════════════════════════════════════════════════════════════

def fig6_pairwise_models() -> None:
    print("Figura 6: matriz pairwise modelos...")

    df_pair = pd.read_csv(TABLE_DIR / "stats_profile_pairwise_attr_models.csv")
    n_attrs = 11

    # Matriz simétrica: (modelo_A, modelo_B) → conteo sig. (por V > 0)
    mat = pd.DataFrame(0, index=MODEL_ORDER, columns=MODEL_ORDER, dtype=float)
    mat_v = pd.DataFrame(np.nan, index=MODEL_ORDER, columns=MODEL_ORDER)

    for _, row in df_pair[df_pair["sig"]].iterrows():
        mat.loc[row["modelo_A"], row["modelo_B"]] += 1
        mat.loc[row["modelo_B"], row["modelo_A"]] += 1
        # Acumular V para promedio
        cur = mat_v.loc[row["modelo_A"], row["modelo_B"]]
        mat_v.loc[row["modelo_A"], row["modelo_B"]] = (
            row["cramers_v"] if pd.isna(cur) else (cur + row["cramers_v"])
        )
        mat_v.loc[row["modelo_B"], row["modelo_A"]] = (
            mat_v.loc[row["modelo_A"], row["modelo_B"]]
        )

    # Normalizar V promedio
    for m1 in MODEL_ORDER:
        for m2 in MODEL_ORDER:
            c = mat.loc[m1, m2]
            if c > 0:
                mat_v.loc[m1, m2] = mat_v.loc[m1, m2] / c

    data = mat.values.astype(float)
    v_data = mat_v.values

    # Panel superior triangular = conteo; inferior triangular = V medio
    fig, ax = plt.subplots(figsize=(8, 7))

    # Máscara: sólo diagonal vacía
    mask_upper = np.triu(np.ones_like(data, dtype=bool), k=1)
    mask_lower = np.tril(np.ones_like(data, dtype=bool), k=-1)

    CMAP_COUNT = LinearSegmentedColormap.from_list(
        "count", ["#FFFFFF", "#FEE0D2", "#FC9272", "#CB181D"]
    )
    CMAP_V = LinearSegmentedColormap.from_list(
        "V", ["#FFFFFF", "#DEEBF7", "#6BAED6", "#08306B"]
    )

    # Fondo general
    ax.imshow(
        np.where(mask_upper, data, np.nan),
        cmap=CMAP_COUNT, vmin=0, vmax=n_attrs, aspect="auto"
    )
    ax.imshow(
        np.where(mask_lower, v_data, np.nan),
        cmap=CMAP_V, vmin=0, vmax=1, aspect="auto"
    )
    # Diagonal en gris
    for i in range(len(MODEL_ORDER)):
        ax.add_patch(mpatches.Rectangle(
            (i - 0.5, i - 0.5), 1, 1,
            color="#EEEEEE", zorder=2
        ))
        ax.text(i, i, MODEL_LABELS[MODEL_ORDER[i]].replace(" ", "\n"),
                ha="center", va="center", fontsize=8.5,
                color="#555555", zorder=3)

    # Anotaciones
    for r in range(len(MODEL_ORDER)):
        for c in range(len(MODEL_ORDER)):
            if r == c:
                continue
            if c > r:  # triángulo superior: conteo
                v = int(data[r, c])
                label = f"{v}/{n_attrs}"
                txt_c = "white" if v >= 7 else "#222222"
                ax.text(c, r, label, ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color=txt_c, zorder=4)
            else:      # triángulo inferior: V medio
                v = v_data[r, c]
                if not np.isnan(v):
                    label = f"V={v:.2f}"
                    txt_c = "white" if v > 0.55 else "#222222"
                    ax.text(c, r, label, ha="center", va="center",
                            fontsize=10, color=txt_c, zorder=4)

    labels = [MODEL_LABELS[m] for m in MODEL_ORDER]
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.tick_params(left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Leyendas manuales
    top_label = (
        "Triángulo superior: nº atributos con distribución\n"
        f"significativamente distinta (de {n_attrs} posibles)"
    )
    bot_label = "Triángulo inferior: Cramér's V medio (efecto)"
    ax.text(0.97, 0.97, top_label, transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, color="#555555",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75, ec="none"))
    ax.text(0.03, 0.03, bot_label, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8.5, color="#555555",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75, ec="none"))

    ax.set_title(
        "Diferencias significativas entre pares de modelos\n"
        "en la distribución de atributos del perfil del paciente",
        fontsize=12, fontweight="bold", pad=12
    )
    fig.tight_layout()
    _save(fig, "fig6_pairwise_models.png")


# ── Main ──────────────────────────────────────────────────────────────────────

_ALL_FIGS = {
    1: fig1_leak_heatmap,
    2: fig2_leak_por_pregunta,
    3: fig3_gender_bias,
    4: fig4_effect_sizes,
    5: fig5_profile_stability,
    6: fig6_pairwise_models,
}

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--fig" in args:
        idx = args.index("--fig") + 1
        selected = [int(x) for x in args[idx:] if x.isdigit()]
    else:
        selected = list(_ALL_FIGS.keys())

    print(f"Generando {len(selected)} figura(s)...\n")
    for n in selected:
        if n in _ALL_FIGS:
            _ALL_FIGS[n]()
        else:
            print(f"  [!] Figura {n} no existe")

    print(f"\nFiguras guardadas en: {FIG_DIR}")

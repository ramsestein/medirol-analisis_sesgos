"""
src/human_eval.py
=================
Herramienta de anotación humana para validar las etiquetas del juez híbrido.

Por defecto muestra solo los casos `needs_review=True` (jueces en desacuerdo
con el regex).  Con --all muestra todas las filas válidas.

Controles:
    1   → REVELA    (revelacion o revelacion_hedged)
    0   → NO REVELA (negacion o evasion)
    s   → SALTAR    (no anota, pasa al siguiente)
    q   → SALIR     (guarda y sale)

Progreso guardado en data/human_labels.csv después de cada anotación.
Al relanzar, retoma donde lo dejó.

Uso:
    python -m src.human_eval
    python -m src.human_eval --all
    python -m src.human_eval --limit 100
"""

from __future__ import annotations

import os
import pathlib
import sys
from datetime import datetime, timezone

import pandas as pd

ROOT       = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"
LABELS_CSV = DATA_DIR / "human_labels.csv"

REVELA     = {"revelacion", "revelacion_hedged"}
NO_REVELA  = {"negacion", "evasion"}


def _fmt_label(label: str | None) -> str:
    """Etiqueta con indicador visual."""
    if pd.isna(label) or label is None:
        return "  ???   "
    label = str(label)
    if label in REVELA:
        return f"[REVELA:{label}]"
    if label in NO_REVELA:
        return f"[NREVELA:{label}]"
    return f"[{label}]"


def _separator(char: str = "─", width: int = 72) -> str:
    return char * width


def show_row(row: pd.Series, idx_disp: int, total: int) -> None:
    """Imprime la fila de forma legible en el terminal."""
    print()
    print(_separator("═"))
    print(f"  [{idx_disp}/{total}]  "
          f"Enfermedad: {row.get('enfermedad','?')}  |  "
          f"P{int(row['pregunta_id'])}  |  "
          f"Ronda {int(row['ronda']) if not pd.isna(row['ronda']) else '?'}  |  "
          f"Modelo: {row.get('modelo','?')}  |  "
          f"Prompt: {row.get('prompt_label','?')}")
    print(_separator())
    print(f"  PREGUNTA : {row['pregunta']}")
    print(_separator())
    # Respuesta: truncar si es muy larga
    resp = str(row["respuesta_final"])
    if len(resp) > 600:
        resp = resp[:597] + "..."
    # Mostrar con indent
    for line in resp.splitlines():
        print(f"  {line}")
    print(_separator())
    print(f"  REGEX  : {_fmt_label(row.get('label'))}"
          f"  (conf={float(row['confidence']):.2f})" if not pd.isna(row.get('confidence')) else
          f"  REGEX  : {_fmt_label(row.get('label'))}")
    print(f"  GPT-5.4: {_fmt_label(row.get('j1_label'))}"
          f"  (conf={float(row['j1_confidence']):.2f})" if not pd.isna(row.get('j1_confidence')) else
          f"  GPT-5.4: {_fmt_label(row.get('j1_label'))}")
    print(f"  DEEPSK : {_fmt_label(row.get('j2_label'))}"
          f"  (conf={float(row['j2_confidence']):.2f})" if not pd.isna(row.get('j2_confidence')) else
          f"  DEEPSK : {_fmt_label(row.get('j2_label'))}")
    print(f"  FINAL  : {_fmt_label(row.get('final_label'))}  "
          f"[{row.get('agreement_type','?')}]"
          + ("  *** NEEDS REVIEW ***" if row.get("needs_review") else ""))
    print(_separator("─"))


def load_existing() -> set[int]:
    """Devuelve el conjunto de índices ya anotados."""
    if not LABELS_CSV.exists():
        return set()
    try:
        df = pd.read_csv(LABELS_CSV, dtype={"df_index": int})
        return set(df["df_index"].tolist())
    except Exception:
        return set()


def append_label(df_index: int, human_label: int, row: pd.Series) -> None:
    """Añade una fila al CSV de anotaciones (append incremental)."""
    record = {
        "df_index":      df_index,
        "human_label":   human_label,
        "human_revela":  1 if human_label == 1 else 0,
        "caso_archivo":  row.get("caso_archivo", ""),
        "enfermedad":    row.get("enfermedad", ""),
        "pregunta_id":   int(row["pregunta_id"]) if not pd.isna(row.get("pregunta_id")) else -1,
        "pregunta":      row.get("pregunta", ""),
        "ronda":         int(row["ronda"]) if not pd.isna(row.get("ronda")) else -1,
        "modelo":        row.get("modelo", ""),
        "prompt_label":  row.get("prompt_label", ""),
        "label_regex":   row.get("label", ""),
        "label_j1":      row.get("j1_label", ""),
        "label_j2":      row.get("j2_label", ""),
        "final_label":   row.get("final_label", ""),
        "agreement_type": row.get("agreement_type", ""),
        "needs_review":  bool(row.get("needs_review", False)),
        "annotated_at":  datetime.now(timezone.utc).isoformat(),
    }
    df_rec = pd.DataFrame([record])
    write_header = not LABELS_CSV.exists()
    df_rec.to_csv(LABELS_CSV, mode="a", index=False, header=write_header,
                  encoding="utf-8-sig")


def run(show_all: bool = False, limit: int | None = None) -> None:
    judged_path = DATA_DIR / "judged_df.parquet"
    if not judged_path.exists():
        print("ERROR: data/judged_df.parquet no encontrado. Ejecuta judge.py primero.")
        sys.exit(1)

    df = pd.read_parquet(judged_path)
    valid_mask = ~df["error"].eq(True) & df["ronda"].notna() & df["final_label"].notna()

    if show_all:
        work = df[valid_mask].copy()
    else:
        work = df[valid_mask & df["needs_review"].eq(True)].copy()

    if limit:
        work = work.head(limit)

    already_done = load_existing()
    pending = work[~work.index.isin(already_done)]

    total     = len(pending)
    annotated = 0

    print(_separator("═"))
    print(f"  Evaluación humana de etiquetas")
    print(f"  Modo: {'TODOS' if show_all else 'needs_review=True'}")
    print(f"  Pendientes: {total}  |  Ya anotados: {len(already_done)}")
    print(f"  Controles:  1=REVELA   0=NO REVELA   s=SALTAR   q=SALIR")
    print(_separator("═"))

    if total == 0:
        print("  ¡No hay casos pendientes!")
        return

    for df_index, row in pending.iterrows():
        show_row(row, annotated + 1, total)

        while True:
            try:
                choice = input("  Tu decisión [1/0/s/q]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  Interrumpido. Progreso guardado.")
                sys.exit(0)

            if choice == "q":
                print(f"\n  Saliendo. Anotados en esta sesión: {annotated}")
                print(f"  Total acumulado: {len(already_done) + annotated}")
                sys.exit(0)
            elif choice == "s":
                print("  [SALTADO]")
                break
            elif choice in ("1", "0"):
                human_label = int(choice)
                append_label(df_index, human_label, row)
                annotated += 1
                already_done.add(df_index)
                tag = "REVELA" if human_label == 1 else "NO REVELA"
                print(f"  ✓ Guardado: {tag}  ({annotated}/{total})")
                break
            else:
                print("  Por favor escribe 1, 0, s o q.")

    print(_separator("═"))
    print(f"  Sesión completada. Anotados: {annotated}/{total}")
    print(f"  Archivo: {LABELS_CSV}")

    # Resumen rápido de concordancia con el sistema
    if annotated > 0 and LABELS_CSV.exists():
        hl = pd.read_csv(LABELS_CSV, encoding="utf-8-sig")
        hl["system_revela"] = hl["final_label"].isin(["revelacion", "revelacion_hedged"]).astype(int)
        agree = (hl["human_label"] == hl["system_revela"]).mean()
        print(f"\n  Concordancia acumulada humano vs. sistema: {agree*100:.1f}%  "
              f"(n={len(hl)})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluación humana de etiquetas MediRol")
    parser.add_argument("--all",   action="store_true",
                        help="Mostrar todos los casos válidos (no solo needs_review)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limitar a N casos")
    args = parser.parse_args()

    run(show_all=args.all, limit=args.limit)

"""
Processing for the emr_cycle_event table.

Turns the raw event export into a per-cycle "responder type" dataset:
    - keeps IVF / egg-retrieval cycles (has STIM; drops IUI/TIC/FET/ET and day-1 LMP)
    - tidies a few mislabeled rows
    - classifies each STIM cycle as 'normal', '?', or 'poor'

Use it from a notebook like this:

    from processing import load_cycle_events, profile, LABEL_COL, ID_COL
    response_df = load_cycle_events()

Nothing runs on import — the file is only read when you call load_cycle_events().
"""

from pathlib import Path
import pandas as pd

# --- column names -----------------------------------------------------------
ID_COL     = "cycleid"     # groups rows into one cycle
LABEL_COL  = "label"       # LMP / STIM / IUI / TRG / RET / ...
DAYNUM_COL = "daynum"      # cycle day number
DATE_COL   = "eventdate"   # event date

# --- where the raw export lives ---------------------------------------------
# Machine-specific path. If you move the data or run on another computer,
# change this one line (or pass a different path to load_cycle_events()).
DATA_PATH = Path(
    r"C:\Users\tamar.schaap\OneDrive - RPSD\project documents and materials"
    r"\Datasets - nAble - For Tamar - UCSD - Confidential PHI\Datasets\emr_cycle_event.csv"
)

# cycles containing any of these labels are not IVF/retrieval cycles -> drop
DROP_LABELS = ["IUI", "TIC", "FET", "ET"]

# obvious data-entry typos -> the label they should be
RELABEL = {
    "DAY 11 TRG": "TRG",
    "DAY 8 TRG":  "TRG",
    "DAY 13 DAY": "TRG",
    "DAY 12 DAY": "TRG",
}

# label glossary (for reference):
#   LMP  Last menstrual period      STIM Ovarian stimulation
#   IUI  Intrauterine insemination  TRG  Trigger
#   RET  Egg retrieval              FET  Frozen embryo transfer
#   TIC  Timed intercourse          ET   Embryo transfer (fresh)
#   AFC  Antral follicle count


def profile(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per column: how full it is, how many distinct values, and its dtype."""
    out = pd.DataFrame({
        "non_null": frame.notna().sum(),
        "nulls":    frame.isna().sum(),
        "distinct": frame.nunique(dropna=True),
        "dtype":    frame.dtypes.astype(str),
    })
    out["pct_null"] = (out["nulls"] / len(frame) * 100).round(1)
    return out.sort_values("non_null", ascending=False)


def load_cycle_events(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load and clean emr_cycle_event.

    Returns one row per STIM cycle with columns
    [cycleid, label, eventdate, response_type], where response_type is
    'normal' (has RET), '?' (has TRG but no RET), or 'poor' (neither).
    """
    df = pd.read_csv(path)

    # normalized labels for matching (strip whitespace, uppercase)
    labels = df[LABEL_COL].astype(str).str.strip().str.upper()

    # --- keep only IVF / egg-retrieval cycles -------------------------------
    # drop any cycle that has a disqualifying label anywhere
    drop_cycles = set(df.loc[labels.isin(DROP_LABELS), ID_COL])

    # also drop cycles with an LMP on day 1: a real stim/retrieval cycle has
    # its LMP earlier (negative day), so a day-1 LMP signals a non-retrieval cycle
    lmp_day1 = (labels == "LMP") & (df[DAYNUM_COL] == 1)
    drop_cycles |= set(df.loc[lmp_day1, ID_COL])

    # keep cycles that have STIM and aren't disqualified (removal wins)
    stim_cycles = set(df.loc[labels == "STIM", ID_COL])
    keep_ids = stim_cycles - drop_cycles
    df = df[df[ID_COL].isin(keep_ids)].copy()

    # --- tidy mislabeled rows -----------------------------------------------
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.upper().replace(RELABEL)

    # --- classify each cycle's response type --------------------------------
    norm = df[LABEL_COL]   # already normalized just above
    flags = pd.DataFrame({
        ID_COL:    df[ID_COL],
        "is_stim": norm == "STIM",
        "is_trg":  norm == "TRG",
        "is_ret":  norm == "RET",
    })
    per_cycle = flags.groupby(ID_COL)[["is_stim", "is_trg", "is_ret"]].any()

    per_cycle["response_type"] = pd.NA
    stim = per_cycle["is_stim"]
    per_cycle.loc[stim &  per_cycle["is_ret"],                        "response_type"] = "normal"
    per_cycle.loc[stim &  per_cycle["is_trg"] & ~per_cycle["is_ret"], "response_type"] = "?"
    per_cycle.loc[stim & ~per_cycle["is_trg"] & ~per_cycle["is_ret"], "response_type"] = "poor"

    df["response_type"] = df[ID_COL].map(per_cycle["response_type"])

    # --- one row per STIM cycle for downstream joins ------------------------
    response_df = (
        df.loc[df[LABEL_COL] == "STIM", [ID_COL, LABEL_COL, DATE_COL, "response_type"]]
          .drop_duplicates()
          .reset_index(drop=True)
    )
    return response_df


# ---------------------------------------------------------------------------
# When you're ready, add the emr_cycle cleaning here the same way:
#
# def load_cycles(path: str | Path = CYCLE_DATA_PATH) -> pd.DataFrame:
#     df = pd.read_csv(path)
#     ...your emr_cycle cleaning logic...
#     return df
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # quick manual check: run `python scripts/processing.py` from the repo root
    out = load_cycle_events()
    print(f"{len(out):,} STIM cycles")
    print(out["response_type"].value_counts(dropna=False))

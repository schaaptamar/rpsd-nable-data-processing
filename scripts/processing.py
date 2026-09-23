"""
Processing for the nAble EMR tables.

Two loaders, each returning a cleaned dataframe (nothing is written to disk):

    load_cycle_events()  -> one row per STIM cycle, with a 'response_type'
                            of 'normal', '?', or 'poor'
    load_cycles()        -> cleaned emr_cycle, one row per cycle

Use them from a notebook like this:

    from processing import load_cycle_events, load_cycles, profile
    response_df = load_cycle_events()
    cycle_df    = load_cycles()

Nothing runs on import — files are only read when you call a loader.
"""

from pathlib import Path
import pandas as pd

# --- column names -----------------------------------------------------------
ID_COL     = "cycleid"     # groups event rows into one cycle
LABEL_COL  = "label"       # LMP / STIM / IUI / TRG / RET / ...
DAYNUM_COL = "daynum"      # cycle day number
DATE_COL   = "eventdate"   # event date

# --- where the raw exports live ---------------------------------------------
# Machine-specific path for the event table. If you move the data or run on
# another computer, change this one line (or pass a path to load_cycle_events()).
CYCLE_EVENT_DATA_PATH = Path("../data/emr_cycle_event.csv")

# emr_cycle export — anchored to the repo's data/ folder so it resolves no
# matter which directory the notebook/kernel happens to run from.
# Assumes layout: repo/scripts/processing.py and repo/data/emr_cycle.csv
CYCLE_DATA_PATH = Path("../data/emr_cycle.csv")

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
    """One row per column: how full it is, how many distinct values, its dtype,
    and whether all non-null values are identical."""
    out = pd.DataFrame({
        "non_null": frame.notna().sum(),
        "nulls": frame.isna().sum(),
        "distinct": frame.nunique(dropna=True),
        "dtype": frame.dtypes.astype(str),
    })

    out["pct_null"] = (out["nulls"] / len(frame) * 100).round(1)

    # True if all non-null values in the column are the same
    out["all_values_same"] = out["distinct"] <= 1

    return out.sort_values("non_null", ascending=False)


def load_cycle_events(path: str | Path = CYCLE_EVENT_DATA_PATH) -> pd.DataFrame:
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

    # Adding Retrieval date
    # now isolating the cases that have label == "RET"
    df_ret = df.loc[df[LABEL_COL] == "RET", [ID_COL, LABEL_COL, "eventdate"]].drop_duplicates()

    # renaming column "retrieval_date" to override eventdate of the df_ret
    df_ret = df_ret.rename(columns={"eventdate": "retrieval_date"})

    #dropping label and response_type columns from df_ret
    df_ret = df_ret.drop(columns=[LABEL_COL])

    # now combining df_ret with response_type by cycleid
    df_combined = pd.merge(response_df, df_ret, on=ID_COL, how="left")

    # Adding Trigger date
    # now isolating the cases that have label == "TRG"
    df_trg = df.loc[df[LABEL_COL] == "TRG", [ID_COL, LABEL_COL, "eventdate"]].drop_duplicates()

    # renaming column "trigger_date" to override eventdate of the df_ret
    df_trg = df_trg.rename(columns={"eventdate": "trigger_date"})

    #dropping label and response_type columns from df_ret
    df_trg = df_trg.drop(columns=[LABEL_COL])

    # now combining df_ret with response_type by cycleid
    df_combined = pd.merge(df_combined, df_trg, on=ID_COL, how="left")
    
    return df_combined


def load_cycles(path: str | Path = CYCLE_DATA_PATH) -> pd.DataFrame:
    """Load and clean emr_cycle (one row per cycle).

    Drops empty and single-value columns, parses dates, and tidies types.
    The join keys id / patid / facility are left untouched so merges stay clean.
    Returns the cleaned dataframe; nothing is written to disk.
    """
    df = pd.read_csv(path)

    prof = profile(df)

    # --- drop columns with no information ----------------------------------
    # dropping columns with all the same value
    cols_to_drop = prof.index[
        (prof["non_null"] == 0) | (prof["all_values_same"])
    ].tolist()

    print(f"Dropping {len(cols_to_drop)} empty or constant columns:\n")
    for c in cols_to_drop:
        print("  -", c)

    df = df.drop(columns=cols_to_drop)

    # --- dropping less informative columns ---------------------------------
    keep_cols = [
        "id", "sart_id", "patid", "partnerid", "barcode", "chloe_id", "height", "weight", "bmi", 
        "addedby", "status", "closed", "cycletype", "schedulestart", "cyclestart", "cycleend", 
        "scheduleend", "event_labels", "doctor", "plan_treatment", "main_note", "art_reason", 
        "cycle_name", "cyclecoordinator", "icsi_tech", "icsi_witness", "insemination_tech", 
        "hist_smoker", "hist_pat_surg_sterile", "hist_monthsattempting", "manual_culturestart",
        "planned_start_date", "planned_retrieval_date", "orig_scheduleend", "edd", "calc_basedate", 
        "calc_type", "last_prospective_report_id", "last_final_report_id", "pgt_reason", "pgt_type", 
        "last_final_report_date", "eggsource_eligibility", "spermsource_eligibility", 
        "planned_eggsource", "planned_spermsource", "end_reason", "icsi_reason", 
        ]
    # filter the dataframe to keep only those columns
    df = df[keep_cols]

    return df


if __name__ == "__main__":
    # quick manual check: run `python scripts/processing.py` from the repo root
    response_df = load_cycle_events()
    print(f"{len(response_df):,} STIM cycles")
    print(response_df["response_type"].value_counts(dropna=False))

    cycle_df = load_cycles()
    print(f"\nemr_cycle: {cycle_df.shape[0]:,} rows x {cycle_df.shape[1]} columns")

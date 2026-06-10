from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw'
OUT_DIR = ROOT / 'data' / 'cleaned'


def normalize_column_name(col_name: str) -> str:
    if not isinstance(col_name, str):
        col_name = str(col_name)
    col_name = col_name.strip().lower()
    col_name = re.sub(r'[^a-z0-9]+', '_', col_name)
    col_name = re.sub(r'_+', '_', col_name).strip('_')
    return col_name


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [normalize_column_name(c) for c in cleaned.columns]
    for col in cleaned.columns:
        cleaned[col] = cleaned[col].astype(str).str.strip()
        cleaned[col] = cleaned[col].replace({'nan': pd.NA, 'none': pd.NA, '': pd.NA})
    return cleaned


def process_files() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_DIR.exists():
        print(f'Raw directory not found: {RAW_DIR}')
        return
    for csv_path in sorted(RAW_DIR.glob('*.csv')):
        df = pd.read_csv(csv_path)
        cleaned = clean_dataframe(df)
        output = OUT_DIR / f'{csv_path.stem}_cleaned.csv'
        cleaned.to_csv(output, index=False)
        print(f'Processed {csv_path.name}')


if __name__ == '__main__':
    process_files()

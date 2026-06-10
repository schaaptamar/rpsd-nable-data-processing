from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEANED_DIR = ROOT / 'data' / 'cleaned'
OUT_FILE = ROOT / 'docs' / 'data_dictionary.csv'


def build_dictionary() -> None:
    rows = []
    for csv_path in sorted(CLEANED_DIR.glob('*_cleaned.csv')):
        df = pd.read_csv(csv_path)
        for col in df.columns:
            rows.append({
                'source_file': csv_path.name,
                'column_name': col,
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isna().sum()),
            })
    pd.DataFrame(rows).to_csv(OUT_FILE, index=False)
    print(f'Saved data dictionary to {OUT_FILE}')


if __name__ == '__main__':
    build_dictionary()

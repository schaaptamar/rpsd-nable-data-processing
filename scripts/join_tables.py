from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEANED_DIR = ROOT / 'data' / 'cleaned'
FINAL_DIR = ROOT / 'data' / 'final'


def join_tables() -> None:
    files = sorted(CLEANED_DIR.glob('*_cleaned.csv'))
    if len(files) < 2:
        raise FileNotFoundError('At least two cleaned files are needed to join tables')
    left = pd.read_csv(files[0])
    right = pd.read_csv(files[1])
    join_key = next((c for c in ['id', 'customer_id', 'account_id'] if c in left.columns and c in right.columns), None)
    if join_key is None:
        raise KeyError('No compatible join key found in the cleaned files')
    merged = left.merge(right, on=join_key, how='left')
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(FINAL_DIR / 'final_dataset.csv', index=False)
    print('Saved joined dataset')


if __name__ == '__main__':
    join_tables()

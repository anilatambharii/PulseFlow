import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input parquet file not found at {input_path}")

    print(f"Preprocessing data from {input_path}...")
    df = pd.read_parquet(input_path)

    # Identify numeric columns
    num_cols = df.select_dtypes(include='number').columns.tolist()

    if 'target' in num_cols:
        num_cols.remove('target')

    print(f"Scaling columns: {num_cols}")
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
    return df

if __name__ == "__main__":
    INPUT_FILE = 'data/intermediate.parquet'
    OUTPUT_FILE = 'data/processed.parquet'

    preprocess_data(INPUT_FILE, OUTPUT_FILE)

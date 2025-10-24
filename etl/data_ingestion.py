import os
import pandas as pd

def load_data(source_path: str, output_path: str):
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source data not found at {source_path}")

    print(f"Loading data from {source_path}...")
    df = pd.read_csv(source_path)
    print(f"Loaded {len(df)} records.")

    # Basic cleaning: drop duplicates and NaNs
    df = df.drop_duplicates().dropna()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"Data saved to {output_path}")
    return df

if __name__ == "__main__":
    INPUT_FILE = 'data/sample.csv'
    OUTPUT_FILE = 'data/intermediate.parquet'
    os.makedirs('data', exist_ok=True)

    # Generate synthetic data if sample.csv doesn't exist
    if not os.path.exists(INPUT_FILE):
        print("sample.csv not found — generating mock dataset...")
        sample_df = pd.DataFrame({
            'feature1': range(100),
            'feature2': [x * 1.5 for x in range(100)],
            'target': [x * 3 + 5 for x in range(100)]
        })
        sample_df.to_csv(INPUT_FILE, index=False)

    load_data(INPUT_FILE, OUTPUT_FILE)

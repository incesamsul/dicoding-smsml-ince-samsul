from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
raw_dir = ROOT / 'breast_cancer_raw'
out_dir = Path(__file__).resolve().parent / 'breast_cancer_preprocessing'
raw_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)

data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df.to_csv(raw_dir / 'breast_cancer_dataset.csv', index=False)
X = df.drop(columns=['target'])
y = df['target']
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
processed = X_scaled.copy()
processed['target'] = y.values
processed.to_csv(out_dir / 'breast_cancer_preprocessing.csv', index=False)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
for name, frame in {'X_train.csv': X_train, 'X_test.csv': X_test, 'y_train.csv': y_train.to_frame(), 'y_test.csv': y_test.to_frame()}.items():
    frame.to_csv(out_dir / name, index=False)
print(f'Preprocessing selesai: {out_dir}')

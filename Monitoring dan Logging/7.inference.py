from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / 'Membangun_model' / 'model_tuned.joblib'
DATA_PATH = Path(__file__).resolve().parents[1] / 'Membangun_model' / 'breast_cancer_preprocessing.csv'
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['target'])
print(model.predict(X.head(5)))

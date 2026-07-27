from app.preprocessing.sessionizer import Sessionizer

sessionizer = Sessionizer()

df = sessionizer.process(
    structured_csv="backend/processed/sample_1000.log_structured.csv",
    label_csv="backend/sample_logs/anomaly_label.csv",
    output_csv="backend/processed/sequences.csv",
)

print("\nShape:")
print(df.shape)

print("\nFirst 10 rows:")
print(df.head(10))

print("\nLabel distribution:")
print(df["Label"].value_counts(dropna=False))

print("\nExample sequence:")
print(df.iloc[0]["EventSequence"])
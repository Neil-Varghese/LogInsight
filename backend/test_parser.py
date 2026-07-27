from app.preprocessing.parser import HDFSParser
import pandas as pd

parser = HDFSParser()

structured_file = parser.parse(
    input_dir="backend/sample_logs",
    output_dir="backend/processed",
    max_lines=1000,      # Development mode
    force=True,
)

print("\n" + "=" * 70)
print("STRUCTURED FILE")
print("=" * 70)
print(structured_file)

df = pd.read_csv(structured_file)

print("\n" + "=" * 70)
print("DATASET SHAPE")
print("=" * 70)
print(df.shape)

print("\n" + "=" * 70)
print("COLUMNS")
print("=" * 70)
print(df.columns.tolist())

print("\n" + "=" * 70)
print("FIRST 10 ROWS")
print("=" * 70)
print(df.head(10))

print("\n" + "=" * 70)
print("UNIQUE EVENT TEMPLATES")
print("=" * 70)
print(df["EventTemplate"].nunique())

print("\n" + "=" * 70)
print("FIRST 20 UNIQUE EVENT TEMPLATES")
print("=" * 70)

for i, template in enumerate(df["EventTemplate"].drop_duplicates().head(20), 1):
    print(f"{i}. {template}")

print("\n" + "=" * 70)
print("EVENT ID COUNTS")
print("=" * 70)
print(df["EventId"].value_counts().head(20))
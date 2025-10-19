import pandas as pd

df = pd.read_csv("data/final_main_dataset.tsv", sep="\t")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(10))

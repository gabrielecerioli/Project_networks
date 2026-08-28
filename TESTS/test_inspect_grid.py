import pandas as pd
import numpy as np

grid_dir = "../data/electrical_grids/GridKit ENTSO-E"

df_buses = pd.read_csv(f"{grid_dir}/buses.csv")
print("BUSES SHAPE:", df_buses.shape)
print("BUSES COLUMNS:", df_buses.columns.tolist())
print("\nBUSES HEAD:")
print(df_buses.head(10))

df_lines = pd.read_csv(f"{grid_dir}/lines.csv")
print("\n" + "="*60)
print("LINES SHAPE:", df_lines.shape)
print("LINES COLUMNS:", df_lines.columns.tolist())
print("\nLINES HEAD:")
print(df_lines.head(10))

print("\n" + "="*60)
print("BUSES INFO:")
print(df_buses.info())
print("\nLINES INFO:")
print(df_lines.info())
print("\nVoltage value counts in lines:")
print(df_lines['voltage'].value_counts().head(10))

import csv
import pandas as pd
import numpy as np

grid_dir = "../data/electrical_grids/GridKit ENTSO-E"

# Test reading buses.csv with csv.reader
buses_rows = []
with open(f"{grid_dir}/buses.csv", "r", encoding="utf-8", errors="replace") as f:
    reader = csv.reader(f)
    header = next(reader)
    print("buses header:", header)
    for i, row in enumerate(reader):
        if len(row) != len(header):
            pass
        buses_rows.append(row)

print(f"Total buses rows read: {len(buses_rows)}")

# Let's inspect how pandas or csv parses lines.csv
lines_rows = []
with open(f"{grid_dir}/lines.csv", "r", encoding="utf-8", errors="replace") as f:
    reader = csv.reader(f)
    lines_header = next(reader)
    print("lines header:", lines_header)
    for i, row in enumerate(reader):
        lines_rows.append(row)

print(f"Total lines rows read: {len(lines_rows)}")

# Build DataFrames
df_buses = pd.DataFrame(buses_rows, columns=header)
df_lines = pd.DataFrame(lines_rows, columns=lines_header)

print("\n--- df_buses sample ---")
print(df_buses.head(5))
print("\n--- df_lines sample ---")
print(df_lines.head(5))

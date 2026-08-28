import pandas as pd

baci_dir = "../data/World_commercial_network/BACI_2017_2024_dataset"

# 1. Read country_codes_V202601.csv
df_cc = pd.read_csv(f"{baci_dir}/country_codes_V202601.csv")
print("country_codes columns:", df_cc.columns.tolist())
print(df_cc.head(5))

# 2. Read Readme.txt
with open(f"{baci_dir}/Readme.txt", "r") as f:
    print("\nReadme.txt:")
    print(f.read())

# 3. Read sample rows of BACI_HS17_Y2024_V202601.csv
df_sample = pd.read_csv(f"{baci_dir}/BACI_HS17_Y2024_V202601.csv", nrows=5)
print("\nBACI 2024 sample:")
print("Columns:", df_sample.columns.tolist())
print(df_sample)

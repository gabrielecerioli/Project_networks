import pandas as pd
import numpy as np

gravity_file = "../data/World_commercial_network/Gravity_V202211.csv"
cols = ['year', 'tradeflow_baci', 'tradeflow_comtrade_o', 'tradeflow_comtrade_d', 'tradeflow_imf_o', 'tradeflow_imf_d']

print("Checking trade flow presence per decade...")
year_stats = {}

for chunk in pd.read_csv(gravity_file, usecols=cols, chunksize=1_000_000):
    for yr, group in chunk.groupby('year'):
        if yr not in year_stats:
            year_stats[yr] = {'baci': 0, 'comtrade': 0, 'imf': 0, 'total': 0}
        year_stats[yr]['baci'] += group['tradeflow_baci'].notna().sum()
        year_stats[yr]['comtrade'] += (group['tradeflow_comtrade_o'].notna() | group['tradeflow_comtrade_d'].notna()).sum()
        year_stats[yr]['imf'] += (group['tradeflow_imf_o'].notna() | group['tradeflow_imf_d'].notna()).sum()
        year_stats[yr]['total'] += len(group)

df_stats = pd.DataFrame.from_dict(year_stats, orient='index')
df_stats.index.name = 'year'
df_stats.reset_index(inplace=True)
df_stats.sort_values('year', inplace=True)

# Print summary every 5 years
print(df_stats[(df_stats['year'] % 5 == 0) | (df_stats['year'] >= 2018)].to_string(index=False))

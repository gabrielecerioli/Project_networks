import networkx as nx
import pandas as pd
import numpy as np

grid_dir = "../data/electrical_grids/GridKit ENTSO-E"

# Re-use parser
from test_power_grid_model import parse_buses, parse_lines

df_buses = parse_buses(f"{grid_dir}/buses.csv")
df_lines = parse_lines(f"{grid_dir}/lines.csv")

valid_buses = set(df_buses.dropna(subset=['lat', 'lon'])['bus_id'])
df_lines = df_lines[df_lines['bus0'].isin(valid_buses) & df_lines['bus1'].isin(valid_buses) & (df_lines['bus0'] != df_lines['bus1'])].copy()

G = nx.Graph()
for _, row in df_lines.iterrows():
    u, v, w = int(row['bus0']), int(row['bus1']), float(row['voltage']) if not np.isnan(row['voltage']) else 220.0
    if G.has_edge(u, v):
        G[u][v]['weight'] += w
    else:
        G.add_edge(u, v, weight=w)

components = sorted(list(nx.connected_components(G)), key=len, reverse=True)
print(f"Top 10 Connected Components sizes:")
for i, comp in enumerate(components[:10], 1):
    print(f"  Component {i:2d}: {len(comp)} nodes")

# Check geographic center / bounds of top components
bus_dict = df_buses.set_index('bus_id')[['lat', 'lon']].to_dict('index')
for i, comp in enumerate(components[:5], 1):
    lats = [bus_dict[n]['lat'] for n in comp]
    lons = [bus_dict[n]['lon'] for n in comp]
    print(f"Component {i:2d} ({len(comp)} nodes): Lat range [{min(lats):.2f}, {max(lats):.2f}], Lon range [{min(lons):.2f}, {max(lons):.2f}]")

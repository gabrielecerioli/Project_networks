import time, torch, numpy as np, pandas as pd, networkx as nx, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

device = torch.device("cpu")
torch.manual_seed(0); np.random.seed(0)

grid_dir = "../data/electrical_grids/GridKit ENTSO-E"

def parse_buses(filepath):
    records = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) == 9:
                bus_id, station_id, voltage, dc, symbol, under_const, tags, x, y = parts
            else:
                bus_id, station_id, voltage, dc, symbol, under_const = parts[:6]
                x, y = parts[-2], parts[-1]
                tags = ",".join(parts[6:-2])
            try:
                records.append({
                    "bus_id": int(bus_id),
                    "voltage": float(voltage) if voltage else np.nan,
                    "under_construction": under_const == 't',
                    "lon": float(x),
                    "lat": float(y)
                })
            except ValueError:
                continue
    return pd.DataFrame(records)

def parse_lines(filepath):
    records = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            geom_idx = line.rfind(",'LINESTRING")
            if geom_idx == -1:
                geom_idx = line.rfind(",LINESTRING")
            if geom_idx != -1:
                front = line[:geom_idx]
                parts = front.split(",")
            else:
                parts = line.split(",")
            if len(parts) >= 8:
                try:
                    line_id, bus0, bus1, voltage, circuits, length, underground, under_const = parts[:8]
                    records.append({
                        "line_id": int(line_id),
                        "bus0": int(bus0),
                        "bus1": int(bus1),
                        "voltage": float(voltage) if voltage else np.nan,
                        "circuits": int(circuits) if circuits and circuits.isdigit() else 1,
                        "length_m": float(length) if length else np.nan,
                        "under_construction": under_const.lower() == 'true'
                    })
                except ValueError:
                    continue
    return pd.DataFrame(records)

print("Loading and parsing ENTSO-E European Power Grid...")
df_buses = parse_buses(f"{grid_dir}/buses.csv")
df_lines = parse_lines(f"{grid_dir}/lines.csv")

# Filter out lines and buses under construction
df_buses = df_buses[~df_buses['under_construction']].copy()
df_lines = df_lines[~df_lines['under_construction']].copy()

# Valid buses with coordinates
valid_buses = set(df_buses.dropna(subset=['lat', 'lon'])['bus_id'])
df_lines = df_lines[df_lines['bus0'].isin(valid_buses) & df_lines['bus1'].isin(valid_buses) & (df_lines['bus0'] != df_lines['bus1'])].copy()

# Line transmission capacity weight: W_edge = circuits * (voltage / 100)^2 (or circuits * voltage)
# Fill missing voltage with median
med_v = df_lines['voltage'].median() if not df_lines['voltage'].isna().all() else 220.0
df_lines['voltage_filled'] = df_lines['voltage'].fillna(med_v)
df_lines['weight'] = df_lines['circuits'] * (df_lines['voltage_filled'] / 100.0)

print(f"Buses: {len(df_buses)}, Active Lines: {len(df_lines)}")

# Build NetworkX Graph to find Giant Connected Component (GCC)
G = nx.Graph()
for _, row in df_lines.iterrows():
    u, v, w = int(row['bus0']), int(row['bus1']), float(row['weight'])
    if G.has_edge(u, v):
        G[u][v]['weight'] += w
    else:
        G.add_edge(u, v, weight=w)

print(f"Full Graph Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
components = list(nx.connected_components(G))
print(f"Connected components count: {len(components)}")
gcc_nodes = max(components, key=len)
print(f"Giant Connected Component (GCC) Nodes: {len(gcc_nodes)} ({len(gcc_nodes)/G.number_of_nodes()*100:.1f}%)")

G_gcc = G.subgraph(gcc_nodes).copy()
print(f"GCC Nodes: {G_gcc.number_of_nodes()}, GCC Edges: {G_gcc.number_of_edges()}")

# If GCC is large (e.g. ~5,000 nodes), let's inspect density and shortest distances
nodes_list = sorted(list(G_gcc.nodes()))
bus_dict = df_buses.set_index('bus_id')[['lat', 'lon']].to_dict('index')

lat_arr = np.array([bus_dict[n]['lat'] for n in nodes_list])
lon_arr = np.array([bus_dict[n]['lon'] for n in nodes_list])

# Filter geographically to European continent bounding box if needed (e.g. lat [34, 72], lon [-15, 45])
in_europe = (lat_arr >= 34.0) & (lat_arr <= 72.0) & (lon_arr >= -15.0) & (lon_arr <= 45.0)
print(f"Nodes in European bounding box: {in_europe.sum()} / {len(nodes_list)}")

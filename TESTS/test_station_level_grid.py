import networkx as nx
import pandas as pd
import numpy as np

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
                    "station_id": int(station_id) if station_id and station_id.isdigit() else station_id,
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

df_buses = parse_buses(f"{grid_dir}/buses.csv")
df_lines = parse_lines(f"{grid_dir}/lines.csv")

print(f"Total buses: {len(df_buses)}, Unique station_ids: {df_buses['station_id'].nunique()}")

bus_to_station = dict(zip(df_buses['bus_id'], df_buses['station_id']))

station_coords = df_buses.groupby('station_id')[['lat', 'lon']].mean().reset_index()
station_to_lat = dict(zip(station_coords['station_id'], station_coords['lat']))
station_to_lon = dict(zip(station_coords['station_id'], station_coords['lon']))

df_lines['st0'] = df_lines['bus0'].map(bus_to_station)
df_lines['st1'] = df_lines['bus1'].map(bus_to_station)

df_lines_st = df_lines.dropna(subset=['st0', 'st1']).copy()
df_lines_st = df_lines_st[df_lines_st['st0'] != df_lines_st['st1']].copy()

print(f"Station-level lines (between distinct substations): {len(df_lines_st)}")

G_st = nx.Graph()
for _, row in df_lines_st.iterrows():
    u, v = row['st0'], row['st1']
    v_val = float(row['voltage']) if not np.isnan(row['voltage']) else 220.0
    c_val = int(row['circuits']) if not np.isnan(row['circuits']) else 1
    w = c_val * (v_val / 100.0) # weighted transmission capacity
    if G_st.has_edge(u, v):
        G_st[u][v]['weight'] += w
    else:
        G_st.add_edge(u, v, weight=w)

print(f"Station Graph: {G_st.number_of_nodes()} substations, {G_st.number_of_edges()} transmission corridors")
components = sorted(list(nx.connected_components(G_st)), key=len, reverse=True)
print(f"Top 5 Substation Connected Components:")
for i, comp in enumerate(components[:5], 1):
    lats = [station_to_lat[s] for s in comp if s in station_to_lat]
    lons = [station_to_lon[s] for s in comp if s in station_to_lon]
    print(f"  Component {i:2d}: {len(comp)} substations ({len(comp)/G_st.number_of_nodes()*100:.1f}%) | Lat [{min(lats):.1f}, {max(lats):.1f}], Lon [{min(lons):.1f}, {max(lons):.1f}]")

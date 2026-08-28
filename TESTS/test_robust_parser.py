import pandas as pd
import numpy as np

grid_dir = "../data/electrical_grids/GridKit ENTSO-E"

def parse_buses(filepath):
    records = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split(",")
        for line_num, line in enumerate(f, 2):
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
            records.append({
                "bus_id": int(bus_id),
                "station_id": station_id,
                "voltage": float(voltage) if voltage else np.nan,
                "dc": dc == 't',
                "symbol": symbol,
                "under_construction": under_const == 't',
                "tags": tags,
                "x": float(x), # lon
                "y": float(y)  # lat
            })
    return pd.DataFrame(records)

def parse_lines(filepath):
    records = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split(",")
        for line_num, line in enumerate(f, 2):
            line = line.strip()
            if not line:
                continue
            # Find LINESTRING or last part
            # geometry is usually 'LINESTRING(...)' at the end
            geom_idx = line.rfind(",'LINESTRING")
            if geom_idx == -1:
                geom_idx = line.rfind(",LINESTRING")
            
            if geom_idx != -1:
                geom = line[geom_idx+1:].strip("'")
                front = line[:geom_idx]
                parts = front.split(",")
                if len(parts) >= 8:
                    line_id, bus0, bus1, voltage, circuits, length, underground, under_const = parts[:8]
                    tags = ",".join(parts[8:]) if len(parts) > 8 else ""
                else:
                    continue
            else:
                parts = line.split(",")
                line_id, bus0, bus1, voltage, circuits, length, underground, under_const = parts[:8]
                geom = parts[-1]
                tags = ",".join(parts[8:-1])
                
            records.append({
                "line_id": int(line_id),
                "bus0": int(bus0),
                "bus1": int(bus1),
                "voltage": float(voltage) if voltage else np.nan,
                "circuits": int(circuits) if circuits and circuits.isdigit() else 1,
                "length": float(length) if length else np.nan,
                "underground": underground.lower() == 'true',
                "under_construction": under_const.lower() == 'true',
                "tags": tags,
                "geometry": geom
            })
    return pd.DataFrame(records)

print("Parsing buses.csv...")
df_buses = parse_buses(f"{grid_dir}/buses.csv")
print(f"Parsed {len(df_buses)} buses successfully.")
print(df_buses.head(5))

print("\nParsing lines.csv...")
df_lines = parse_lines(f"{grid_dir}/lines.csv")
print(f"Parsed {len(df_lines)} lines successfully.")
print(df_lines.head(5))

import pandas as pd
import os

# Path to your downloaded telemetry file
telemetry_file = "data/rwl_tel_hr_bihar_999_2021_2025.csv"

print("Reading telemetry file (this may take a moment)...")
df = pd.read_csv(telemetry_file, low_memory=False)

# Extract unique stations with their district, river, latitude, longitude
# Use the column names exactly as they appear in the file
# From the HTML we saw columns: Station, District, River, Latitude, Longitude
stations = df[['Station', 'District', 'River', 'Latitude', 'Longitude']].drop_duplicates('Station')
# Remove rows with missing coordinates
stations = stations.dropna(subset=['Latitude', 'Longitude'])

# Save to CSV
stations.to_csv('data/station_coords.csv', index=False)
print(f"✅ Created station_coords.csv with {len(stations)} unique stations.")
print("Sample stations:")
print(stations.head())
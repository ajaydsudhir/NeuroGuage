import pandas as pd
import OpenMuse

# Define a variable for the file path
file_path = "data/trial1/trial1.txt"

# Read the data from the text file
with open(file_path, "r", encoding="utf-8") as f:
    messages = f.readlines()

# Decode the raw data using OpenMuse
data = OpenMuse.decode_rawdata(messages)

# Save EEG, ACCGYRO, and OPTICS datasets to separate CSV files
for key in ['EEG', 'ACCGYRO', 'OPTICS']:
    if key in data and isinstance(data[key], pd.DataFrame):
        output_file = f"{file_path}_{key.lower()}.csv"
        data[key].to_csv(output_file, index=False)
        print(f"Saved {key} data to {output_file}")
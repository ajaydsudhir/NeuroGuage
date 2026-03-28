import pandas as pd
import OpenMuse 
import matplotlib.pyplot as plt

# Define a variable for the file path
file_path = "data/trial1/trial1.txt"

# Read the data from the text file
with open(file_path, "r", encoding="utf-8") as f:
    messages = f.readlines()
data = OpenMuse.decode_rawdata(messages)

# Plot Movement Data
data["ACCGYRO"].plot(
    x="time",
    y=["ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z"],
    subplots=True,
    title='Movement Data'
)
plt.show()

# Plot EEG Data
data['EEG'].plot(
    x='time',
    y=['EEG_TP9', 'EEG_AF7', 'EEG_AF8', 'EEG_TP10', 'AUX_1', 'AUX_2', 'AUX_3', 'AUX_4'],
    subplots=True,
    title='EEG Data'
)
plt.show()

# Plot OPTICS Data
data['OPTICS'].plot(
    x='time',
    y=['OPTICS_LO_NIR', 'OPTICS_RO_NIR', 'OPTICS_LO_IR', 'OPTICS_RO_IR', 'OPTICS_LI_NIR', 'OPTICS_RI_NIR', 'OPTICS_LI_IR', 'OPTICS_RI_IR', 'OPTICS_LO_RED', 'OPTICS_RO_RED', 'OPTICS_LO_AMB', 'OPTICS_RO_AMB', 'OPTICS_LI_RED', 'OPTICS_RI_RED', 'OPTICS_LI_AMB', 'OPTICS_RI_AMB'],
    subplots=True,
    title='OPTICS Data'
)
plt.show()
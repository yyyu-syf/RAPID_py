import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone

# -------------------------
# Step 1: Read the Output NetCDF File
# -------------------------

# Specify the path to your output NetCDF file
output_file = '../rapid_data/Qout_San_Guad_20100101_20131231_VIC0125_3H_utc.nc'
# Open the output NetCDF file
dataset = netCDF4.Dataset(output_file, 'r')

# Access the variables
Qout = dataset.variables['Qout'][:]      # Routed discharge values (time, rivid)
time = dataset.variables['time'][:]      # Time variable
rivid = dataset.variables['rivid'][:]    # River IDs

# Convert time units to datetime objects
time_units = dataset.variables['time'].units
time_calendar = dataset.variables['time'].calendar if 'calendar' in dataset.variables['time'].ncattrs() else 'standard'
time_values_cftime = netCDF4.num2date(time, units=time_units, calendar=time_calendar)

# Function to convert cftime to datetime
def cftime_to_datetime(cftime_dates):
    datetime_dates = []
    for t in cftime_dates:
        try:
            datetime_dates.append(datetime(t.year, t.month, t.day, t.hour, t.minute, t.second))
        except ValueError:
            # Handle dates outside the valid range of datetime.datetime
            datetime_dates.append(None)
    return datetime_dates

time_values = cftime_to_datetime(time_values_cftime)

# Convert time_values to numpy array
time_values = np.array(time_values)

# Create valid_indices where time_values is not None
valid_indices = time_values != np.array(None)

# Alternatively, use list comprehension
# valid_indices = [t is not None for t in time_values]

# Filter time_values and Qout to include only valid entries
time_values = time_values[valid_indices]
Qout = Qout[valid_indices, :]

# -------------------------
# Step 2: Select the Data to Plot
# -------------------------

# Select the river reach to plot (e.g., the first reach)
reach_index = 0  # Index in the array (0-based index)
reach_id = rivid[reach_index]

# Get the discharge time series for the selected reach
Qout_reach = Qout[:, reach_index]

# -------------------------
# Step 3: Plot the Data
# -------------------------

# Create a figure and plot
plt.figure(figsize=(12, 6))
plt.plot(time_values, Qout_reach, label=f'River Reach ID {reach_id}')
plt.xlabel('Time')
plt.ylabel('Discharge (m³/s)')
plt.title('River Discharge Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Rotate the x-axis labels for better readability
plt.gcf().autofmt_xdate()

# Display the plot
plt.show()

# -------------------------
# Step 4: Close the Dataset
# -------------------------

# Close the NetCDF dataset
dataset.close()

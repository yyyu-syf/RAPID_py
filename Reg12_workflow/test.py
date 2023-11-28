import netCDF4 as nc
import pandas as pd

# Open the NetCDF file
dataset = nc.Dataset('m3_riv_NLDAS_MOS0125_H.A20120101.3hrly.002.nc')

# Create an empty DataFrame
df = pd.DataFrame()


print(dataset.variables.keys())
print(dataset.variables['m3_riv'])
print(dataset.variables['COMID'])


# Extract 'm3_riv' data and save to CSV
m3_riv_data = dataset.variables['m3_riv'][:]
m3_riv_df = pd.DataFrame(m3_riv_data)
m3_riv_df.to_csv('m3_riv.csv', index=False)

# Extract 'COMID' data and save to CSV
comid_data = dataset.variables['COMID'][:]
comid_df = pd.DataFrame(comid_data)
comid_df.to_csv('COMID.csv', index=False)

# Close the dataset
dataset.close()


import netCDF4 as nc
import pandas as pd

# Open the NetCDF file
dataset = nc.Dataset('./rapid_data/m3_riv_San_Guad_20100101_20131231_VIC0125_3H_utc_err_R286_D_scl.nc')
# dataset = nc.Dataset('./rapid_data/m3_riv_San_Guad_20100101_20131231_VIC0125_M_utc.nc')
# dataset = nc.Dataset('./rapid_data/m3_riv_San_Guad_20100101_20131231_ENS0125_M_utc.nc')
dataset = nc.Dataset('./rapid_data/m3_riv_San_Guad_20100101_20131231_VIC0125_3H_utc.nc')

# Create an empty DataFrame
df = pd.DataFrame()


print(dataset.variables.keys())
print(dataset.variables['m3_riv'])
print(dataset.variables['rivid'])
# print(dataset.variables['m3_riv_err'])


# Extract 'm3_riv' data and save to CSV
m3_riv_data = dataset.variables['m3_riv'][:]
m3_riv_df = pd.DataFrame(m3_riv_data)
m3_riv_df.to_csv('./rapid_data/m3_riv.csv', index=False)

# Extract 'rivid' data and save to CSV
comid_data = dataset.variables['rivid'][:]
comid_df = pd.DataFrame(comid_data)
comid_df.to_csv('./rapid_data/rivid.csv', index=False)

# # Extract 'm3_riv_err' data and save to CSV
# comid_data = dataset.variables['m3_riv_err'][:]
# comid_df = pd.DataFrame(comid_data)
# comid_df.to_csv('./rapid_data/m3_riv_err.csv', index=False)

# # Close the dataset
dataset.close()


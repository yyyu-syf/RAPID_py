import geopandas as gpd
import matplotlib.pyplot as plt
# Attempt to load and plot the river network again now that all necessary files are uploaded

# File paths for the shapefile components
shp_path = './rapid_data/NHDFlowline_San_Guad/NHDFlowline_San_Guad.shp'
dbf_path = './rapid_data/NHDFlowline_San_Guad/NHDFlowline_San_Guad.dbf'
shx_path = './rapid_data/NHDFlowline_San_Guad/NHDFlowline_San_Guad.shx'
prj_path = './rapid_data/NHDFlowline_San_Guad/NHDFlowline_San_Guad.prj'

# Load the shapefile
river_network = gpd.read_file(shp_path)

# Plot the river network
# plt.figure(figsize=(10, 10))
river_network.plot()
plt.title('River Network')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

save_path = './river_network_plot.png'
plt.savefig(save_path)
plt.show()

# File paths for the new shapefile components
shp_path_2 = './rapid_data/StreamGageEvent_San_Guad_comid_withdir_full_2010_2013/StreamGageEvent_San_Guad_comid_withdir_full_2010_2013.shp'
dbf_path_2 = './rapid_data/StreamGageEvent_San_Guad_comid_withdir_full_2010_2013/StreamGageEvent_San_Guad_comid_withdir_full_2010_2013.dbf'
shx_path_2 = './rapid_data/StreamGageEvent_San_Guad_comid_withdir_full_2010_2013/StreamGageEvent_San_Guad_comid_withdir_full_2010_2013.shx'
prj_path_2 = './rapid_data/StreamGageEvent_San_Guad_comid_withdir_full_2010_2013/StreamGageEvent_San_Guad_comid_withdir_full_2010_2013.prj'

# Load the shapefile
stream_gage_event = gpd.read_file(shp_path_2)

# Plot the river network
# plt.figure(figsize=(10, 10))
stream_gage_event.plot()
plt.title('Stream Gage Event Network')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Save the figure
save_path_2 = './stream_gage_event_plot.png'
plt.savefig(save_path_2)
plt.show()

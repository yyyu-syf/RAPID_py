from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import math
import matplotlib.animation as animation
import geopandas as gpd

# Earth's radius in meters (mean radius)
R = 6371000

def lat_lon_to_cartesian(lat, lon, lat_ref, lon_ref):
    """
    Convert latitude and longitude to Cartesian coordinates.
    """
    
    x = R * (lon - lon_ref) * math.cos(math.radians(lat_ref))
    y = R * (lat - lat_ref)
    return x, y


file_path = './rapid_data/NHDFlowline_San_Guad/reach_info.csv'
kf_path = './model_saved/Discharge_est.csv'
# kf_path = './model_saved/u.csv'
kf_id_path = './rapid_data/rivid.csv'


kf_data = pd.read_csv(kf_path, header=None)
kf_id = pd.read_csv(kf_id_path, header=None).to_numpy()
# column_names = ['id']
# kf_id.columns = column_names
widths  = kf_data.values

river_data = pd.read_csv(file_path)
rivid = river_data['Reach ID']
first_lat = river_data['Start Latitude']
first_long = river_data['Start Longitude']
last_lat = river_data['End Latitude']
last_long = river_data['End Longitude']
lengths = river_data['Length (km)'].values

# Normalize the widths by their daily maximum 
normalized_widths = widths / np.max(widths, axis=0) * 5
normalized_widths = widths 

# # Above is only for testing,This is corrent:
# normalized_widths = widths / np.max(widths)

ref_lat = first_lat[0]
ref_long = first_long[0]


first_vertex_cartesian = [lat_lon_to_cartesian(lat, lon, ref_lat, ref_long) for lat, lon in zip(first_lat, first_long)]
last_vertex_cartesian = [lat_lon_to_cartesian(lat, lon, ref_lat, ref_long) for lat, lon in zip(last_lat, last_long)]

first_vertex_x_cartesian, first_vertex_y_cartesian = zip(*first_vertex_cartesian)
last_vertex_x_cartesian, last_vertex_y_cartesian = zip(*last_vertex_cartesian)


def update_real_dimensions(frame):
    plt.clf()
    num_reaches = len(first_vertex_x_cartesian) # from shp file
    num_kf_reaches = len(kf_id) # from kf estimation
    num_widths = normalized_widths.shape[1]
    
    # import time
    # time.sleep(1100)

    for i in range(num_kf_reaches):
        # find the corresponding index from shp id
        id_kf = int(kf_id[i])
        id_shp_condition = river_data['Reach ID'] == id_kf
        if id_shp_condition.any():
            id_shp = river_data.index[id_shp_condition].tolist()[0]
            
            # Calculate the direction vector of the line segment in Cartesian coordinates
            dir_x = last_vertex_x_cartesian[id_shp] - first_vertex_x_cartesian[id_shp]
            dir_y = last_vertex_y_cartesian[id_shp] - first_vertex_y_cartesian[id_shp]

            # Normalize the direction vector
            length = np.sqrt(dir_x**2 + dir_y**2)
            dir_x /= length
            dir_y /= length

            # Calculate the width for the current day, proportional to the length of the river reach
            if normalized_widths[frame, i] > 1e+03:
                normalized_widths[frame, i] = 1e+03

            width = normalized_widths[frame, i] * lengths[id_shp]
            
            # print(f"id_shp {id_shp} | id_kf = {id_kf} | i = {i} | width {width}")
            ### Plot reaches with actual width
            # Calculate the offset for the width
            offset_x = -dir_y * width / 2
            offset_y = dir_x * width / 2

            # Calculate the coordinates of the vertices of the rectangle representing the river reach in Cartesian coordinates
            x1 = first_vertex_x_cartesian[id_shp] + offset_x
            y1 = first_vertex_y_cartesian[id_shp] + offset_y
            x2 = last_vertex_x_cartesian[id_shp] + offset_x
            y2 = last_vertex_y_cartesian[id_shp] + offset_y
            x3 = last_vertex_x_cartesian[id_shp] - offset_x
            y3 = last_vertex_y_cartesian[id_shp] - offset_y
            x4 = first_vertex_x_cartesian[id_shp] - offset_x
            y4 = first_vertex_y_cartesian[id_shp] - offset_y

            # Plot the rectangle
            plt.fill([x1, x2, x3, x4], [y1, y2, y3, y4], 'b-')

    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('River Width on Day {} (Real Dimensions)'.format(frame + 1))
    plt.grid(True)
    plt.axis('equal')

# Create an animation with real dimensions
fig = plt.figure(figsize=(10, 10))
ani_real_dimensions = FuncAnimation(fig, update_real_dimensions, frames=365, repeat=True)
video_path = "./model_saved/river_width_changes.mp4"
# plt.show()

# ani_real_dimensions.save(video_path, writer="ffmpeg", fps=15)
ani_real_dimensions.save('./model_saved/animation.gif', writer='pillow', fps=10)

# FFwriter = animation.FFMpegWriter(fps=10)
# ani_real_dimensions.save(video_path, writer = FFwriter)

# FFwriter = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)

# # Save the animation
# ani_real_dimensions.save(video_path, writer = FFwriter)

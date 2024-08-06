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
# kf_path = './model_saved/discharge_est_kf2.csv'
# kf_path = './model_saved/river_lateral_est.csv'
kf_path = './model_saved/discharge_est.csv'
# kf_path = './model_saved/open_loop_river_lateral_est.csv'
# kf_path = './model_saved/river_lateral_est.csv'
# kf_path = './model_saved/u.csv'
kf_id_path = './rapid_data/riv_bas_id_San_Guad_hydroseq.csv'


kf_data = pd.read_csv(kf_path, header=None)
kf_id = pd.read_csv(kf_id_path, header=None).to_numpy()

print(kf_data[0].shape)
cutoff = 5175
kf_data = kf_data[0:cutoff]
kf_id = kf_id[0:cutoff]
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

print(f"value.shape {widths.shape}")

# # Above is only for testing,This is correct:
# normalized_widths = widths / np.max(widths, axis=0) * 5
normalized_widths = widths
# normalized_widths_safe = np.where(normalized_widths == 0, 1e-10, normalized_widths)
# normalized_widths = np.log2(normalized_widths_safe)
print(f"max widths { np.max(normalized_widths)} | index {np.argmax(np.array(normalized_widths))}")
normalized_widths = np.cbrt(normalized_widths / np.max(normalized_widths) *50)

for frame in range(normalized_widths.shape[0]):
    for i in range(normalized_widths.shape[1]):
        if normalized_widths[frame, i] < 0.01 :
            normalized_widths[frame, i] = 00.005
        if normalized_widths[frame, i] > 1e+10:
            normalized_widths[frame, i] = 1e+10
            
            
print(f"max widths after normalization { np.max(normalized_widths)}")
            

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
            # print(f"delta {lengths[id_shp] - length}")
            # # length = 3e1
            # dir_x /= length
            # dir_y /= length

            # width = normalized_widths[frame, i] * lengths[id_shp]
            width = normalized_widths[frame, i]
            
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
            # plt.text((x1 + x3) / 2, (y1 + y3) / 2, str(id_kf), color='red', fontsize=8, ha='center', va='center')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('River Width on Day {} (Real Dimensions)'.format(frame + 1))
    plt.grid(True)
    plt.axis('equal')

# Create an animation with real dimensions
fig = plt.figure(figsize=(10, 10))
ani_real_dimensions = FuncAnimation(fig, update_real_dimensions, frames=50, repeat=True)
# video_path = "./model_saved/river_width_changes.mp4"
plt.show()

# ani_real_dimensions.save(video_path, writer="ffmpeg", fps=15)
# ani_real_dimensions.save('./model_saved/dkf_river_state.gif', writer='pillow', fps=10)
# ani_real_dimensions.save('./model_saved/kf2_river_state.gif', writer='pillow', fps=10)
# ani_real_dimensions.save('./model_saved/kf_river_state.gif', writer='pillow', fps=3)

# FFwriter = animation.FFMpegWriter(fps=10)
# ani_real_dimensions.save(video_path, writer = FFwriter)

# FFwriter = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)

# # Save the animation
# ani_real_dimensions.save(video_path, writer = FFwriter)

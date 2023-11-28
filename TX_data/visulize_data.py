from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import math
import matplotlib.animation as animation
import geopandas as gpd

# Replace 'your_shapefile.shp' with the path to your .shp file
file_path = 'TX_width.shp'

# Read the shapefile
gdf = gpd.read_file(file_path)
river_shp_data = pd.read_csv(file_path)

river_shp_data['LINKNO'] = river_shp_data['LINKNO'].apply(ast.literal_eval)
river_shp_data['DSLINKNO'] = river_shp_data['DSLINKNO'].apply(ast.literal_eval)
river_shp_data['USLINKNO1'] = river_shp_data['USLINKNO1'].apply(ast.literal_eval)
river_shp_data['USLINKNO2'] = river_shp_data['USLINKNO2'].apply(ast.literal_eval)

# Display the first few rows of the geodataframe
print(gdf.head())



# Read the CSV file
file_path = 'river_info.csv'
river_data = pd.read_csv(file_path)

# Convert string representations of tuples to actual tuples
river_data['first vertex'] = river_data['first vertex'].apply(ast.literal_eval)
river_data['last vertex'] = river_data['last vertex'].apply(ast.literal_eval)

# Extract x and y coordinates of the first and last vertices
first_vertex_x, first_vertex_y = zip(*river_data['first vertex'])
last_vertex_x, last_vertex_y = zip(*river_data['last vertex'])

# Plot the river reaches
plt.figure(figsize=(10, 10))
plt.plot([first_vertex_x, last_vertex_x], [first_vertex_y, last_vertex_y], 'k-')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('River Reaches')
plt.grid(True)
plt.axis('equal')
plt.show()

# Extract widths for each day
widths = river_data.iloc[:, 4:].values

# Normalize the widths by their daily maximum
normalized_widths = widths / np.max(widths, axis=0)

# Extract the lengths of the river reaches
lengths = river_data['Length'].values

# Earth's radius in meters (mean radius)
R = 6371000

def lat_lon_to_cartesian(lat, lon, lat_ref, lon_ref):
    """
    Convert latitude and longitude to Cartesian coordinates.
    """
    x = R * (lon - lon_ref) * math.cos(math.radians(lat_ref))
    y = R * (lat - lat_ref)
    return x, y

# Convert the first and last vertices to Cartesian coordinates
first_vertex_cartesian = [lat_lon_to_cartesian(lat, lon, first_vertex_y[0], first_vertex_x[0]) for lat, lon in zip(first_vertex_y, first_vertex_x)]
last_vertex_cartesian = [lat_lon_to_cartesian(lat, lon, first_vertex_y[0], first_vertex_x[0]) for lat, lon in zip(last_vertex_y, last_vertex_x)]

first_vertex_x_cartesian, first_vertex_y_cartesian = zip(*first_vertex_cartesian)
last_vertex_x_cartesian, last_vertex_y_cartesian = zip(*last_vertex_cartesian)

# Update the plot function to use Cartesian coordinates and real dimensions
def update_real_dimensions(frame):
    plt.clf()
    for i in range(len(first_vertex_x_cartesian)):
        # Calculate the direction vector of the line segment in Cartesian coordinates
        dir_x = last_vertex_x_cartesian[i] - first_vertex_x_cartesian[i]
        dir_y = last_vertex_y_cartesian[i] - first_vertex_y_cartesian[i]

        # Normalize the direction vector
        length = np.sqrt(dir_x**2 + dir_y**2)
        dir_x /= length
        dir_y /= length

        # Calculate the width for the current day, proportional to the length of the river reach
        width = normalized_widths[i, frame] * lengths[i]

        # Calculate the offset for the width
        offset_x = -dir_y * width / 2
        offset_y = dir_x * width / 2

        # Calculate the coordinates of the vertices of the rectangle representing the river reach in Cartesian coordinates
        x1 = first_vertex_x_cartesian[i] + offset_x
        y1 = first_vertex_y_cartesian[i] + offset_y
        x2 = last_vertex_x_cartesian[i] + offset_x
        y2 = last_vertex_y_cartesian[i] + offset_y
        x3 = last_vertex_x_cartesian[i] - offset_x
        y3 = last_vertex_y_cartesian[i] - offset_y
        x4 = first_vertex_x_cartesian[i] - offset_x
        y4 = first_vertex_y_cartesian[i] - offset_y

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
video_path = "/river_width_changes2.mp4"
# ani_real_dimensions.save(video_path, writer="ffmpeg", fps=15)
ani_real_dimensions.save('animation.gif', writer='pillow', fps=20)
# plt.show()

# FFwriter = animation.FFMpegWriter(fps=10)
# ani_real_dimensions.save(video_path, writer = FFwriter)

# FFwriter = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)

# # Save the animation
# ani_real_dimensions.save(video_path, writer = FFwriter)


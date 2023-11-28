"""

Shapefile, also known as ESRI Shapefile, is a popular geospatial vector data format used by Geographic Information Systems (GIS) software to store location, shape, and attributes of geographic features such as points, lines, and polygons.

A shapefile is typically made up of multiple files with the same base filename, but with different extensions. These files include:

.shp file: This is the main file that stores the geometric data for the features in the shapefile. It contains information about the shape, size, and location of each feature.

.shx file: This is the shape index file that stores the index of the feature geometry in the .shp file. It allows GIS software to quickly locate and read the features in the .shp file.

.dbf file: This is a dBASE format file that contains the attribute data associated with the features in the shapefile. It stores the attribute data in a tabular format and is linked to the geometry of the features in the .shp file through a unique identifier known as the "record number".

.prj file: This is the projection file that contains the coordinate system and projection information used to represent the geographic features in the shapefile. It is essential for accurate spatial analysis and mapping.

Shapefiles are widely used in GIS applications for mapping, spatial analysis, and data visualization. They are compatible with a wide range of software applications and can be easily shared and distributed.

*Attributes vs Features*

A feature represents a geographic entity or object that can be represented as a point, line, or polygon on a map. Examples of features include roads, rivers, buildings, and parcels of land. A feature can have one or more attributes associated with it that provide additional information about the feature, such as its name, length, area, or population.

An attribute is a characteristic or property of a feature that describes some aspect of the feature. Attributes are stored in the attribute table of a GIS dataset, such as a shapefile or geodatabase feature class. Each row in the attribute table represents a feature, and each column represents an attribute. The attribute values provide information about the feature, such as its name, size, shape, or classification.

In summary, a feature is a representation of a real-world object or entity on a map, and an attribute is a property or characteristic of that feature that provides additional information about it.

About the TX_width dataset:

* 2948 features = stream segments = `LineString` or `MultiLineString` is possible. The `LineString` is specified by a set of Vertices between network nodes. 
* 385 attributes (per feature). The first 19 attributes correspond to various features of the river network topology (e.g. which segments are upstream), stream segment geometry (e.g. slope, length), and segment ID (COMID).


"""
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
import pandas as pd

import matplotlib.pyplot as plt
from geopy.distance import geodesic


import os
import pickle

dir_path = os.path.dirname(os.path.realpath(__file__))

# Define a bounding box
gdf  = gpd.read_file(dir_path + '/TX_width.shp')
# Store the GeoDataFrame as a pickle object
# with open(dir_path + 'TX_width.pickle', 'wb') as f:
#    pickle.dump(gdf, f)
# Load the GeoDataFrame from the pickle file
# with open(dir_path + 'TX_width.pickle', 'rb') as f:
#     gdf = pickle.load(f)

# 
print(gdf)

# Print the geometry type of each feature
print("gdf.geometry.type")
print(gdf.geometry.type) #Length: 2949

print("gdf.geometry")
print(gdf.geometry)


print("gdf.geometry.iloc[0]")
print(gdf.geometry.iloc[0])
geometry = gdf.geometry
iloc0 = gdf.geometry.iloc[0]
# first vertex
# get the first Linestring object in the GeoPandas DataFrame
line = gdf.loc[0, 'geometry']
# access the coordinates of the vertices in the Linestring object
coords = list(line.coords)
print("coords[0]")
print(coords[0])

# Get the metadata
print("gdf.crs.")
print(gdf.crs) #  "EPSG:4326"

print("gdf.total_bounds")
print(gdf.total_bounds) # [-104. 30.00017859 -100. 33.01594744]

print("gdf.shape")
print(gdf.shape) # (2949, 385)

# Get a list of attribute names
attribute_names = list(gdf.columns)
# Print the list of attribute names
print("attributes")
print(*attribute_names, sep = ", ")
print("# attributes = ", len(attribute_names))

# Access the attribute table as a pandas DataFrame
df = gdf[['w1', 'w2']] 
# Print the first few rows of the attribute table
print(df)

#
w1 = gdf.iloc[:,19]
print("w1.head()")
print(w1.head())

print("first feature has the following attributes:")
print(*gdf.iloc[0], sep = ", ")

#print("last feature has the following attributes:")
#print(*gdf.iloc[-1], sep = ", ")

#### Save the required attributes onto a csv file ####
# stream widths
stream_width = gdf.iloc[:,19:365+19]
# lengths
stream_length = gdf[['Length']]
# first vertex, last vertex coordinates
first_vertex = [] 
last_vertex = []
for k in range(0,len(gdf)):
    stream_seg = gdf.loc[k, 'geometry']
    
    if isinstance(stream_seg, MultiLineString):
       first_vertex.append(stream_seg.geoms[0].coords[0]) # first vertex of the 1st line string
       last_vertex.append(stream_seg.geoms[-1].coords[-1]) # last vertex of the last line string
       #_x = gpd.GeoDataFrame(
       #         {'geometry': [stream_seg]})
        #_x.plot()
       #plt.show()
    elif isinstance(stream_seg, LineString):
        first_vertex.append(stream_seg.coords[0]) # first vertex of the 1st line string
        last_vertex.append(stream_seg.coords[-1]) # last vertex of the last line string
        

    else:
        print("Error")


river_info_df = pd.concat([stream_length, stream_width], axis=1)

river_info_df = pd.concat([pd.Series(last_vertex, name='last vertex'), river_info_df], axis=1)
river_info_df = pd.concat([pd.Series(first_vertex, name='first vertex'), river_info_df], axis=1)

# save the Pandas DataFrame to a CSV file
river_info_df.to_csv('river_info.csv', index=True)

# Create a matplotlib figure and axis
fig, ax = plt.subplots()
# Plot the shapefile onto the axis
gdf.plot(ax=ax)
# Display the plot
plt.show()

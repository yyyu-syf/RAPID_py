import geopandas as gpd
import pandas as pd

# Load the shapefile
shapefile_path = './rapid_data/NHDFlowline_San_Guad/NHDFlowline_San_Guad.shp'
gdf = gpd.read_file(shapefile_path)
  
# Function to extract reach information
def extract_reach_info(row):
    # Extracting the start and end points of the geometry
    coords = list(row['geometry'].coords)
    # print(f"Geo info: {row['geometry'].columns}")

    # Display the first few geometries
    print("\nFirst few geometries in the 'geometry' column:")
    print(gdf['geometry'].head()[0])
    print(len(gdf['geometry'].head()[0]))
    

    import time
    time.sleep(100)
    start_point = coords[0]
    end_point = coords[-1]

    # Extracting the latitude and longitude
    start_lat, start_lon = start_point[1], start_point[0]
    end_lat, end_lon = end_point[1], end_point[0]

    return {
        'Reach ID': row['COMID'],
        'Start Latitude': start_lat,
        'Start Longitude': start_lon,
        'End Latitude': end_lat,
        'End Longitude': end_lon,
        'Length (km)': row['LENGTHKM']
    }

# Applying the function to each row in the GeoDataFrame
reach_info = gdf.apply(extract_reach_info, axis=1)
reach_info_df = reach_info.apply(pd.Series)

# Ensure 'Reach ID' is of integer type
reach_info_df['Reach ID'] = reach_info_df['Reach ID'].astype(int)

# Save the DataFrame to a CSV file
csv_output_path = './rapid_data/NHDFlowline_San_Guad/reach_info.csv'
reach_info_df.to_csv(csv_output_path, index=False)


# from osgeo import ogr
# import pandas as pd

# # File paths
# shapefile_path = './rapid_data/NHDFlowline_San_Guad/NHDFlowline_San_Guad.shp'
# dbf_path = './rapid_data/NHDFlowline_San_Guad/NHDFlowline_San_Guad.dbf'

# # Opening the shapefile
# driver = ogr.GetDriverByName("ESRI Shapefile")
# data_source = driver.Open(shapefile_path, 0) # 0 means read-only
# layer = data_source.GetLayer()

# # Initialize lists to store data
# reach_ids = []
# latitudes = []
# longitudes = []
# lengths = []

# # Iterating through the features (reaches) in the shapefile
# for feature in layer:
#     # Extract reach ID
#     reach_id = feature.GetField("COMID")
#     reach_ids.append(reach_id)

#     # Extract geometry of the feature
#     geom = feature.GetGeometryRef()
#     length = geom.Length()
#     lengths.append(length)

#     # Calculate and store the centroid latitude and longitude
#     centroid = geom.Centroid()
#     latitudes.append(centroid.GetY())
#     longitudes.append(centroid.GetX())

# # Create a DataFrame
# df = pd.DataFrame({
#     'Reach ID': reach_ids,
#     'Latitude': latitudes,
#     'Longitude': longitudes,
#     'Length': lengths
# })

# # Exporting the DataFrame to a CSV file
# csv_file_path = './rapid_data/NHDFlowline_San_Guad/reach_data.csv'
# df.to_csv(csv_file_path, index=False)


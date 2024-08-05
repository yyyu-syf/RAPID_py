import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# Load the shapefile
shp_path = "./rapid_data/NHDFlowline_San_Guad/NHDFlowline_San_Guad.shp"
shp_data = gpd.read_file(shp_path)

# Load the discharge data without treating the first row as the header
discharge_data_path = "./model_saved/discharge_est.csv"
discharge_data = pd.read_csv(discharge_data_path, header=None)

# Load the reach ID data without treating the first row as the header
reach_id_data_path = "./rapid_data/riv_bas_id_San_Guad_hydroseq.csv"
reach_id_data = pd.read_csv(reach_id_data_path, header=None)

# Ensure the discharge data columns match the reach IDs
reach_ids = reach_id_data.iloc[:, 0].values
discharge_data.columns = reach_ids

# Combine the discharge data with the shapefile data
for t in range(discharge_data.shape[0]):
    shp_data[f"Q{t+1}"] = discharge_data.iloc[t].reindex(shp_data.COMID).values

# Simulate the strmOrder column
np.random.seed(42)  # For reproducibility
shp_data['strmOrder'] = np.random.randint(1, 5, size=len(shp_data))

# Filter out polylines with stream order > 0
shp_sub = shp_data[shp_data['strmOrder'] > 0]

# Calculate global maximum discharge for color scaling
Qcols = [col for col in shp_sub.columns if col.startswith('Q')]
globQmax = shp_sub[Qcols].max().max()

# Set up color and line width gradients
colfun = plt.cm.viridis
lwdRamp = np.linspace(0.15, 12, 50)

# Plot each time step
for i, Qcol in enumerate(Qcols, start=1):
    fig, ax = plt.subplots(1, 2, figsize=(12, 10), gridspec_kw={'width_ratios': [6, 1]})
    
    # Map plotting
    Q = shp_sub[Qcol]
    norm = plt.Normalize(vmin=0, vmax=globQmax)
    colors = colfun(norm(Q))
    lwds = np.interp(Q, (0, globQmax), (0.15, 12))
    
    shp_sub.plot(ax=ax[0], color=colors, linewidth=lwds)
    ax[0].set_title(f"Discharge at Day {i}")
    
    # Color bar plotting
    sm = plt.cm.ScalarMappable(cmap=colfun, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax[1])
    cbar.set_label("Q (cms)")
    
    plt.show()
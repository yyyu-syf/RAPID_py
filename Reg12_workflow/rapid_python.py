from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import math
import matplotlib.animation as animation
import geopandas as gpd
import os

class RAPIDKF():
    def __init__(self, args, env) -> None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.load_file(dir_path)
            
    def load_file(self,dir_path):
        id_path = dir_path + '/basin_id_Reg12_hydroseq.csv'
        connect_path = dir_path + '/rapid_connect_Reg12.csv'
        m3riv_path = dir_path + '/m3_riv.csv'
        x_path = dir_path + '/x_Reg12_Mosaic_pa2.csv'
        k_path = dir_path + '/k_Reg12_Mosaic_pa2.csv'
        # obs_path = dir_path + '/k_Reg12_Mosaic_pa2.csv'
        
        id_data = pd.read_csv(id_path)
        connect_data = pd.read_csv(connect_path)
        m3riv_data = pd.read_csv(m3riv_path)
        x_data = pd.read_csv(x_path)
        k_data = pd.read_csv(k_path)
        self.length = x_data.shape[0]
        
        ### process lateral inflow from 3-hourly to daily
        lateral_daily = m3riv_data.iloc.sum(axis=0)
        lateral_daily_df = lateral_daily.to_frame()
        print(f"daily inflow shape: {lateral_daily_df}")
        
        ### process Muskinggum parameter
        if x_data.nunique().iloc[0] == 1:
            print("All Muskinggum x are the same:", x_data.iloc[0, 0])
        
            
        
    def calculate_connectivity(self,connect_data):
        # Sort the DataFrame by ID
        # river_network = connect_data.sort_values(by='ID column name')
        river_network = connect_data

        # Initialize the connectivity matrix
        connectivity_matrix = np.zeros((self.length, self.length), dtype=int)

        # Populate the matrix
        for index, row in river_network.iterrows():
            reach_id = row['ID column name']
            # Adjust index if ID does not start at 0 or 1
            for i in range(4, 8):  # Columns 4 to 7
                upstream_id = row[i]
                if upstream_id > 0:  # Check if the upstream ID is not 0
                    # Adjust indices if necessary
                    connectivity_matrix[upstream_id, reach_id] = 1

    def calculate_Cs(self):
        # Calculate the values of C1, C2, and C3
        C1 = (delta_t/2 - k * x) / ((k * (1 - x) + delta_t/2))
        C2 = (delta_t/2 + k * x) / ((k * (1 - x) + delta_t/2))
        C3 = (k * (1 - x) - delta_t/2) / ((k * (1 - x) + delta_t/2))

        return C1, C2, C3
    
    def simulate(self):
        pass
    
    






if __name__ == '__main__':
    

    
        
    k=1
    x=0.35
    delta_t = 3
    C1, C2, C3 = calculate_Cs(k, x, delta_t)
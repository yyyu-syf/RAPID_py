from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import math
import matplotlib.animation as animation
import geopandas as gpd
import os
import time
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

class PreProcessor():
    def __init__(self) -> None:
        pass
        
    def pre_processing(self,**kwargs):
        # Extracting parameters from the dictionary
        id_path = kwargs['id_path'] 
        connect_path = kwargs['connect_path'] 
        m3riv_path = kwargs['m3riv_path'] 
        m3riv_id_path = kwargs['m3riv_id_path'] 
        x_path = kwargs['x_path'] 
        k_path = kwargs['k_path'] 
        obs_path = kwargs['obs_path'] 
        obs_id_path = kwargs['obs_id_path'] 
        vic_model_path = kwargs['vic_model_path'] 
        ens_model_path = kwargs['ens_model_path'] 
        self.days = kwargs['days'] 
        self.month = kwargs['month'] 
        self.radius = kwargs['radius'] 
        self.epsilon = kwargs['epsilon'] 
        self.i_factor = kwargs['i_factor'] 
        
        # Load data
        reach_id = pd.read_csv(id_path, header=None)
        connect_data = pd.read_csv(connect_path, header=None)
        m3riv_data = pd.read_csv(m3riv_path, header=None)
        x_data = pd.read_csv(x_path, header=None)
        k_data = pd.read_csv(k_path, header=None)
        obs_data = pd.read_csv(obs_path, header=None)
        obs_id = pd.read_csv(obs_id_path, header=None)
        vic_data_m = pd.read_csv(vic_model_path, header=None)
        ens_data_m= pd.read_csv(ens_model_path, header=None)
        
        # cutoff data based on perference
        cutoff = len(reach_id)
        reach_id = reach_id[0:cutoff]
        connect_data = connect_data[0:cutoff]
        x_data = x_data[0:cutoff]
        k_data = k_data[0:cutoff]
        vic_data_m = vic_data_m[0:self.month]
        ens_data_m = ens_data_m[0:self.month]
        m3riv_data = m3riv_data[0:self.days*8]
        obs_data = obs_data[0:self.days]
        self.l_reach = x_data.shape[0]
        
        print(f"reach nums: {self.l_reach}")
        print(f"m3riv_data {m3riv_data.shape}")
        print(f"obs_id:  {obs_id.shape}")
        
        ### process lateral inflow from 3-hourly to daily varaged
        lateral_daily = m3riv_data.to_numpy().reshape((self.days, 8, m3riv_data.shape[-1])).sum(axis=1)
        lateral_daily_averaged = lateral_daily/8
        
        print(f"lateral_daily data shape: {lateral_daily.shape} ")
        
        # process Muskinggum parameter
        self.musking_k = np.zeros(self.l_reach)
        self.musking_x = np.zeros(self.l_reach)
        self.musking_C1 = np.zeros(self.l_reach)
        self.musking_C2 = np.zeros(self.l_reach)
        self.musking_C3 = np.zeros(self.l_reach)
        
        if x_data.nunique().iloc[0] == 1:
            self.musking_x = np.full(self.l_reach,x_data.iloc[0, 0])
            print("All Muskinggum x are the same:", x_data.iloc[0, 0])
            
        else:
            for i , x  in enumerate(x_data):
                self.musking_x[i] = x
        
        for i , k  in enumerate(k_data):
            self.musking_k[i] = k
            
        for i in range(len(k_data)):
            k = self.musking_k[i]
            x = self.musking_x[i]
            delta_t = 1
            self.musking_C1[i], self.musking_C2[i], self.musking_C3[i] = self.calculate_Cs(k, x, delta_t)
        
        self.musking_C1 = np.diag(self.musking_C1)
        self.musking_C2 = np.diag(self.musking_C2)
        self.musking_C3 = np.diag(self.musking_C3)
        
        # # Calculate connectivity matrix
        self.calculate_connectivity(connect_data)
        
        # # Calculate dynamics coefficient
        self.calculate_coefficient()
        
        # S
        obs_id = obs_id.to_numpy().flatten()
        reach_id = reach_id.to_numpy().flatten()
        S = np.zeros((len(obs_id),len(reach_id)), dtype=int)
        for i, obs in enumerate(obs_id):
            index = np.where(reach_id == obs)[0]
            S[i, index] = 1
        
        # He = np.dot(S,self.Ae)
        # H0 = np.dot(S,self.A0)
        H = S
        
        # R at initial state
        R0 = 0.1*obs_data.to_numpy()[0]
        R = np.diag(R0**2)
        
        print(f"Dim of R: {R.shape}")
        
        # P and prune P based on radius
        delta = abs((vic_data_m - ens_data_m).sum(axis=0)/12)
        print(f"shape of delta:{delta.shape}")
        P = self.i_factor * np.dot(delta.values.reshape(-1,1),delta.values.reshape(-1,1).T)
        pruned_P = self.pruneP(P,connect_data,self.radius)
        
        print(f"Dim of P: {P.shape}")
        print(f"vic shape: {vic_data_m.shape}")
        
        np.savetxt("model_saved/P_delta.csv", delta, delimiter=",")
        np.savetxt("model_saved/P.csv", P , delimiter=",")
        np.savetxt("model_saved/prunedP.csv", pruned_P , delimiter=",")
        np.savetxt("model_saved/H.csv", H , delimiter=",")
        np.savetxt("model_saved/R.csv", R, delimiter=",")
        np.savetxt("model_saved/z.csv", obs_data, delimiter=",")
        np.savetxt("model_saved/u.csv", lateral_daily_averaged, delimiter=",")
        np.savetxt("model_saved/Ae.csv", self.Ae, delimiter=",")
        np.savetxt("model_saved/A0.csv", self.A0, delimiter=",")
        np.savetxt("model_saved/N.csv", self.N, delimiter=",")
        
        return self.Ae, self.A0, H, pruned_P, R, lateral_daily_averaged, obs_data.to_numpy()
        
        
    def calculate_connectivity(self,connect_data):
        column_names = ['id', 'downId', 'numUp', 'upId1','upId2','upId3','upId4']
        connect_data.columns = column_names
        # Sort the DataFrame by ID
        # river_network = connect_data.sort_values(by='ID column name')
        river_network = connect_data
        # Initialize the connectivity matrix
        connectivity_matrix_N = np.zeros((self.l_reach, self.l_reach), dtype=int)

        # Populate the matrix
        for row_index, row in river_network.iterrows():
            reach_id = row[0]
            # check id:
            upStreamNum = 0
            for i in range(3, 6):  # Columns 4 to 7
                upstream_id = row[i]
                if upstream_id > 0:  # Check if the upstream ID is not 0
                    upStreamNum += 1
                    condition = river_network['id'] == upstream_id
                    if condition.any():
                        up_row_index = river_network.index[condition].tolist()
                        
                    # Adjust indices if necessary
                    connectivity_matrix_N[row_index, up_row_index] = 1

        self.N = connectivity_matrix_N
        

    def calculate_coefficient(self):
        '''
        Coefficient Ae, A0
        '''
        ### (I-C1N)^-1 ###
        mat_I = np.identity(self.l_reach)
        A1 = mat_I - np.dot(self.musking_C1,self.N)
        # print(f"if {np.all(np.triu(A1, k=1) == 0)}")
        A1_inv = np.linalg.inv(A1)
        # A1_sparse = sp.csr_matrix(A1)
        # A1_inv = splinalg.inv(A1_sparse)
        print(A1_inv.shape)
        A1_inv[A1_inv < self.epsilon] = 0
        
        ### C1+C2 ###
        A2 = self.musking_C1 + self.musking_C2
        
        ### C3+C2N ###
        A3 = self.musking_C3 + np.dot(self.musking_C2,self.N)
        
        ### [I-C1N]^-1(C3+C2N)] ###
        A4 = np.dot(A1_inv,A3)
        
        ### [I-C1N]^-1(C1+C2)] ###
        A5 = np.dot(A1_inv,A2)
        
        # ### Ae ###
        print(f"A4 shape {A4.shape}")
        np.savetxt("model_saved/A4.csv", A4[0:100,0:100], delimiter=",")
        print(f"A5 shape {A5.shape}")
        np.savetxt("model_saved/A5.csv", A5[0:100,0:100], delimiter=",")
        
        Ae = np.zeros((self.l_reach,self.l_reach))
        for p in np.arange(0,96):
            # Ae += np.dot((96-p)/96 * A4**p,A5) 
            Ae += np.dot((96-p)/96 * np.linalg.matrix_power(A4, p),A5) 
        
        A0 = np.zeros((self.l_reach,self.l_reach))
        for p in np.arange(1,96):
            # A0 += 1/96 * A4**p 
            A0 += 1/96 * np.linalg.matrix_power(A4, p)
        
        self.Ae = Ae
        self.A0 = A0
        # self.Ae = sum(np.dot((96-p)/96 * A4**p,A5) for p in range(96))
        # self.A0 = sum(1/96 * A4**p for p in np.arange(1,96))
        

    def calculate_Cs(self,k, x, delta_t):
        # Calculate the values of C1, C2, and C3
        C1 = (delta_t/2 - k * x) / ((k * (1 - x) + delta_t/2))
        C2 = (delta_t/2 + k * x) / ((k * (1 - x) + delta_t/2))
        C3 = (k * (1 - x) - delta_t/2) / ((k * (1 - x) + delta_t/2))

        return C1, C2, C3
    
    
    def pruneP(self,P, river_network, radius):
        # Generate mask
        column_names = ['id', 'downId', 'numUp', 'upId1','upId2','upId3','upId4']
        river_network.columns = column_names
        maskP = np.zeros_like(P)
        
        for row_index, row in river_network.iterrows():
            maskP[row_index,row_index] = 1
            
            downStreamId = row[0]
            # print(f" type  {river_network.index[river_network['id'] == upStreamId]}")
            for _ in range(radius):
                condition = river_network['id'] == downStreamId
                if condition.any():
                    down_row_index = river_network.index[condition].tolist()

                maskP[row_index,down_row_index] = 1
                maskP[down_row_index,row_index] = 1
                downStreamId = river_network.iloc[down_row_index]['downId'].tolist()[0]
                
        # save the mask
        w = P.shape[0]
        plt.figure(figsize=(8, 8))
        plt.imshow(maskP, cmap='Greys', interpolation='none')
        # plt.colorbar(label='Density')
        plt.title(f"P density with R = {radius}")

        # Adding axis ticks to show indices
        plt.xticks(ticks=np.arange(0, w, w/10), labels=np.arange(0, w, w/10))
        plt.yticks(ticks=np.arange(0, w, w/10), labels=np.arange(0, w, w/10))
        plt.grid(color='gray', linestyle='-', linewidth=0.5)

        # Saving the plot with coordinates in high-resolution
        plt.savefig("model_saved/density_P.png", dpi=300, bbox_inches='tight')
        
        return P * maskP
            
        
    
    
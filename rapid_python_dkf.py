import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
import geopandas as gpd
import os
import time
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from utility import PreProcessor
import pickle
from infokf import InfoKalmanFilter

class RAPIDKF():
    def __init__(self, load_mode=0) -> None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.epsilon = 0.001 #muksingum parameter threshold
        self.radius = 20
        self.i_factor = 2.58 #enforced on covaraince P
        self.days = 366 #2010 year 366 days
        self.month = self.days//365 * 12
        self.timestep = 0
        if load_mode==0:
            self.load_file(dir_path)
        else:
            self.load_pkl(dir_path)
        
    def load_pkl(self,dir_path):       
        dis_name = 'model_saved/dkf_load_coef.pkl'
        with open(os.path.join(dir_path, dis_name),'rb') as f:
            saved_dict = pickle.load(f)
        
        self.Ae = saved_dict['dkf_Ae'] 
        self.A0 = saved_dict['dkf_A0'] 
        self.S = saved_dict['dkf_H'] 
        self.P = saved_dict['dkf_P'] 
        self.R = saved_dict['dkf_R'] 
        self.u = saved_dict['dkf_u'] 
        self.obs_data = saved_dict['dkf_obs_data'] 
        self.H = np.dot(self.S, self.Ae)
        
        dI = np.eye(self.P.shape[0]) * 0.0001 #avoid singularity 
        self.P += dI        
            
    def load_file(self,dir_path):
        id_path = dir_path + '/rapid_data/riv_bas_id_San_Guad_hydroseq.csv'
        connect_path = dir_path + '/rapid_data/rapid_connect_San_Guad.csv'
        m3riv_path = dir_path + '/rapid_data/m3_riv.csv'
        m3riv_id_path = dir_path + '/rapid_data/m3_riv.csv'
        x_path = dir_path + '/rapid_data/x_San_Guad_2004_1.csv'
        k_path = dir_path + '/rapid_data/k_San_Guad_2004_1.csv'
        obs_path = dir_path + '/rapid_data/Qobs_San_Guad_2010_2013_full.csv'
        obs_id_path = dir_path + '/rapid_data/obs_tot_id_San_Guad_2010_2013_full.csv'
        vic_model_path = dir_path + '/rapid_data/m3_riv_vic_month.csv'
        ens_model_path = dir_path + '/rapid_data/m3_riv_ens_month.csv'
        
        params = {
                'id_path': id_path,
                'connect_path': connect_path,
                'm3riv_path': m3riv_path,
                'm3riv_id_path': m3riv_id_path,
                'x_path': x_path,
                'k_path': k_path,
                'obs_path': obs_path,
                'obs_id_path': obs_id_path,
                'vic_model_path': vic_model_path,
                'ens_model_path': ens_model_path,
                'days': self.days,
                'month': self.month,
                'radius': self.radius,
                'epsilon': self.epsilon,
                'i_factor': self.i_factor,
                
            }
    
        dataProcessor = PreProcessor()
        self.Ae, self.A0, self.S, self.P, self.R, self.u, self.obs_data = dataProcessor.pre_processing(**params)
        
        saved_dict = {
                'dkf_Ae': self.Ae,
                'dkf_A0': self.A0,
                'dkf_H': self.S,
                'dkf_P': self.P,
                'dkf_R': self.R,
                'dkf_u': self.u,
                'dkf_obs_data': self.obs_data,
            }
        
        dis_name = 'model_saved/dkf_load_coef.pkl'
        with open(os.path.join(dir_path, dis_name), 'wb') as f:
                pickle.dump(saved_dict, f)
            
    
    def simulateDKF(self):
        '''
        Simulation of multiple nodes perform decentralized estimation
        '''
        kf_estimation = []
        discharge_estimation = []
        self.x = self.u[0]
        
        infoKF_agents = []
        dim_z_individual = 36
        
        # Generate agents
        infoKF_agents = []
        num_dec_agent = int(self.obs_data.shape[-1]/dim_z_individual)
        
        for i in range(num_dec_agent):
            H_i = self.H[i* dim_z_individual:((i+1)* dim_z_individual),:]
            S_i = self.S[i* dim_z_individual:((i+1)* dim_z_individual),:]
            R_i = self.R[i* dim_z_individual:((i+1)* dim_z_individual),i* dim_z_individual:((i+1)* dim_z_individual)]
            infoKF = InfoKalmanFilter(A = self.Ae, B = self.A0, H = H_i, S = S_i, R = R_i, P = self.P, x0 = self.u[0])
            infoKF_agents.append(infoKF)
            
        # Run simulation
        for timestep in range(self.days):
            print(f"days: {timestep}")
            infoPlist = []
            infoPpredlist = []
            xlist = []
            xpredlist = []
            
            ### Individual estimation
            for i_kf, agent in enumerate(infoKF_agents):
                print(f"[est] id_agents: {i_kf}")
                agent.predict(self.u[timestep])  

                z_i = self.obs_data[timestep][i_kf*dim_z_individual:((i_kf+1)*dim_z_individual)]
                xpred ,Ppred,x_broadcast_,infoP_broadcast_ = agent.updateInfoKF(z_i)
                
                infoPlist.append(infoP_broadcast_)
                infoPpredlist.append(Ppred)
                xlist.append(x_broadcast_)
                xpredlist.append(xpred)
                
                ##
                # agent.update_discharge()
                
            ### Fusion via information exchange
            for i_kf, agent in enumerate(infoKF_agents):
                print(f"[broadcast] id_agents: {i_kf}")
                agent.updateAfterCommunication(infoPlist,
                                                infoPpredlist,
                                                xlist,xpredlist
                                                )
                agent.update_discharge()
                
            # All estimation should be the same after exchange, just extract agent 0th
            kf_estimation.append(infoKF_agents[0].getState()) 
            discharge_estimation.append(infoKF_agents[0].getQ0()) 

        np.savetxt("model_saved/dkf_discharge_est.csv", discharge_estimation, delimiter=",")
        np.savetxt("model_saved/dkf_lateral_est.csv", kf_estimation, delimiter=",")
    
    
if __name__ == '__main__':
    rapid_kf = RAPIDKF(load_mode=1)
    rapid_kf.simulateDKF()

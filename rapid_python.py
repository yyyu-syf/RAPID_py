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
        dis_name = 'load_coef.pkl'
        with open(os.path.join(dir_path, dis_name),'rb') as f:
            saved_dict = pickle.load(f)
        
        self.Ae = saved_dict['Ae'] 
        self.A0 = saved_dict['A0'] 
        self.H = saved_dict['H'] 
        self.P = saved_dict['P'] 
        self.R = saved_dict['R'] 
        self.u = saved_dict['u'] 
        self.obs_data = saved_dict['obs_data'] 
            
            
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
        self.Ae, self.A0, self.H, self.P, self.R, self.u, self.obs_data = dataProcessor.pre_processing(**params)
        
        saved_dict = {
                'Ae': self.Ae,
                'A0': self.A0,
                'H': self.H,
                'P': self.P,
                'R': self.R,
                'u': self.u,
                'obs_data': self.obs_data,
            }
        
        dis_name = 'load_coef.pkl'
        with open(os.path.join(dir_path, dis_name), 'wb') as f:
                pickle.dump(saved_dict, f)
            
            
    def predict(self,u=None):
        if u is None:
            u = np.zeros((self.B.shape[-1], 1))
        self.x_last = self.x
        self.x = u
        # self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        self.timestep += 1
        
        return self.x
    
    
    def update(self, z, inputType=None):
        if inputType is not None:
            self.u, self.u_var = self.input_estimation(z)
            self.x = self.x + np.dot(self.B,self.u)
            innovation=  z - np.dot(self.H, self.x)
        else: 
            innovation = z - np.dot(self.H, self.x)
        
        self.R = np.diag(0.1*z)
        print(f"R shape {self.R}")
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  
        self.x = self.x + np.dot(K, innovation)
        # self.P = self.P - np.dot(np.dot(K,self.H),self.P)   
        
        return 
    
    
    def simulate(self):
        kf_estimation = []
        self.x = self.u[0]
        print(f"state shape: {self.x.shape}")
        for timestep in range(self.days):
            self.predict(self.u[timestep])
            self.update(self.obs_data[timestep])
            kf_estimation.append(self.getState())
    
    
    def getState(self):
        return self.x
    
    def getP(self):
        return self.P

    def getR(self):
        return self.R
    
    def getH(self):
        return self.H
    
    
    
    
if __name__ == '__main__':
    rapid_kf = RAPIDKF(load_mode=1)
    rapid_kf.simulate()
    k=1
    x=0.35
    delta_t = 3
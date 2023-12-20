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

class RAPIDKF():
    def __init__(self) -> None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.epsilon = 0.001 #muksingum parameter threshold
        self.radius = 20
        self.i_factor = 2.58 #enforced on covaraince P
        self.days = 366 #2010 year 366 days
        self.month = self.days//365 * 12
        self.timestep = 0
        self.load_file(dir_path)
        
            
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
        self.Ae, self.A0, self.H, self.P, self.R, self.lateral_daily_averaged = dataProcessor.pre_processing(**params)
        
    
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

        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  
        self.x = self.x + np.dot(K, innovation)
        # self.P = self.P - np.dot(np.dot(K,self.H),self.P)   
        
        return 
    
    
    def simulate(self):
        kf_estimation = []
        for timestep in range(self.days):
            self.predict(self.lateral_daily_averaged[timestep])
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
    rapid_kf = RAPIDKF()
    rapid_kf.simulate()
    k=1
    x=0.35
    delta_t = 3
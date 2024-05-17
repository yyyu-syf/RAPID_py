import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
import geopandas as gpd
import os
from scipy.stats import multivariate_normal as scipy_guassian
import time
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from utility import PreProcessor
import pickle

## The state of this KF: lateral inflow

class RAPIDKF():
    def __init__(self, load_mode=0) -> None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.epsilon = 0.0001 #muksingum parameter threshold
        self.radius = 20
        self.i_factor = 2.58 #enforced on covaraince P
        self.days = 366 #2010 year 366 days
        self.month = self.days//365 * 12
        self.timestep = 0
        if load_mode==0 or load_mode == 2:
            self.load_file(dir_path)
        if load_mode==1 or load_mode == 2:
            self.load_pkl(dir_path)
        
    def load_pkl(self,dir_path):       
        dis_name = 'model_saved/load_coef.pkl'
        with open(os.path.join(dir_path, dis_name),'rb') as f:
            saved_dict = pickle.load(f)
        
        self.Ae = saved_dict['Ae'] 
        self.A0 = saved_dict['A0'] 
        self.A4 = saved_dict['A4'] 
        self.A5 = saved_dict['A5'] 
        self.H1 = saved_dict['H1'] 
        self.H2 = saved_dict['H2'] 
        self.S = saved_dict['S'] 
        self.P = saved_dict['P'] 
        self.R = saved_dict['R'] 
        self.u = saved_dict['u'] 
        self.obs_data = saved_dict['obs_data'] 
        self.H = np.dot(self.S,self.Ae)
        
            
    def load_file(self,dir_path):
        id_path = dir_path + '/rapid_data/rivid.csv'
        id_path_sorted = dir_path + '/rapid_data/riv_bas_id_San_Guad_hydroseq.csv'
        connect_path = dir_path + '/rapid_data/rapid_connect_San_Guad.csv'
        m3riv_path = dir_path + '/rapid_data/m3_riv.csv'
        m3riv_d_path = dir_path + '/rapid_data/m3_d_riv.csv'
        m3riv_id_path = dir_path + '/rapid_data/m3_riv.csv'
        x_path = dir_path + '/rapid_data/x_San_Guad_2004_1.csv'
        k_path = dir_path + '/rapid_data/k_San_Guad_2004_1.csv'
        obs_path = dir_path + '/rapid_data/Qobs_San_Guad_2010_2013_full.csv'
        obs_id_path = dir_path + '/rapid_data/obs_tot_id_San_Guad_2010_2013_full.csv'
        vic_model_path = dir_path + '/rapid_data/m3_riv_vic_month.csv'
        ens_model_path = dir_path + '/rapid_data/m3_riv_ens_month.csv'
        
        params = {
                'id_path': id_path,
                'id_path_sorted': id_path_sorted,
                'connect_path': connect_path,
                'm3riv_path': m3riv_path,
                'm3riv_d_path':m3riv_d_path,
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
        self.Ae, self.A0, self.S, self.P, self.R, self.u, self.obs_data, \
            self.A4, self.A5, self.H1, self.H2 = dataProcessor.pre_processing(**params)
        
        saved_dict = {
                'Ae': self.Ae,
                'A0': self.A0,
                'A4': self.A4,
                'A5': self.A5,
                'H1': self.H1,
                'H2': self.H2,
                'S': self.S,
                'P': self.P,
                'R': self.R,
                'u': self.u,
                'obs_data': self.obs_data,
            }
                
        dis_name = 'model_saved/load_coef.pkl'
        with open(os.path.join(dir_path, dis_name), 'wb') as f:
                pickle.dump(saved_dict, f)
            
    def simulate(self):
        
        kf_estimation = []
        discharge_estimation = []
        open_loop_x = []
        self.x = self.u[0]     #self.x is Qe
        self.x = np.zeros_like(self.u[0])
        
        self.Q0 = np.ones_like(self.u[0]) *10
        # self.Q0 = (self.u[0])
        print(f"state shape: {self.x.shape}")
        print(f"rank of P:{np.linalg.matrix_rank(self.P)}, shape: {self.P.shape}")
        for timestep in range(self.days):
            if timestep <= 0:
                x_predict = self.predict(self.u[timestep])
            else:
                x_predict = self.predict()
                
            # self.update(self.obs_data[timestep])
            self.update_discharge()
            
            kf_estimation.append(self.getState()) 
            discharge_estimation.append(self.getQ0())
            open_loop_x.append(self.getQ0())

        np.savetxt("model_saved/discharge_est.csv", discharge_estimation, delimiter=",")
        np.savetxt("model_saved/river_lateral_est.csv", kf_estimation, delimiter=",")
        np.savetxt("model_saved/open_loop_river_lateral_est.csv", open_loop_x, delimiter=",")
        
    def predict(self,u=None):
        if u is not None:
            self.x = u
        else:
            self.x = np.zeros_like(self.x)
        
        # self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        self.timestep += 1
        return self.x
    
    
    def update(self, z, inputType=None):
        ### In RAPID model, inputType = None
        diag_R = 0.01*z**2
        self.R = np.diag(diag_R)
        
        z = z - np.dot(self.S, np.dot(self.A0,self.Q0))
        if inputType is not None:
            self.u, self.u_var = self.input_estimation(z)
            self.x = self.x + np.dot(self.B,self.u)
            innovation=  z - np.dot(self.H, self.x)
        else: 
            innovation = z - np.dot(self.H, self.x)
        
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        if np.linalg.matrix_rank(S) < S.shape[0]:
            delta_S = 0.001*np.eye(S.shape[0])
            S += delta_S
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  
        self.x = self.x + np.dot(K, innovation)
        # self.P = self.P - np.dot(np.dot(K,self.H),self.P)   

    
    def update_discharge(self):
        # self.Q0 = np.dot(self.Ae,self.x) + np.dot(self.A0,self.Q0)
        # p= 95
        self.Q0 = self.H1 @ self.x + \
                                self.H2 @ self.Q0
        # self.Q0 = np.linalg.matrix_power(self.A4, p) @ self.Q0
                                
    
    def input_estimation(self,z): 
        F = np.dot(self.H, self.B)
        S = np.dot(np.dot(self.H,self.P),self.H.T)+self.R
        M_1 = np.linalg.inv(np.dot(np.dot(F.T,np.linalg.inv(S)),F))
        M_2 = np.dot(F.T,np.linalg.inv(S))
        M = np.dot(M_1,M_2)
        innovation = z - np.dot(self.H, self.x)
        u = np.dot(M,innovation)
        u_var = M_1

        return u, u_var 
    
    def getState(self):
        return self.x
    
    def getP(self):
        return self.P

    def getR(self):
        return self.R
    
    def getH(self):
        return self.H
    
    def getQ0(self):
        return self.Q0
    
    
    
    
if __name__ == '__main__':
    rapid_kf = RAPIDKF(load_mode=1)
    rapid_kf.simulate()
    k=1
    x=0.35
    delta_t = 3
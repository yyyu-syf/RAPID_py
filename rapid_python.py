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
import copy

## The state of this KF: lateral inflow

class RAPIDKF():
    def __init__(self, load_mode=0) -> None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.epsilon = 0.1 #muksingum parameter threshold
        self.radius = 10
        self.i_factor = 2.58 #enforced on covaraince P
        self.days = 366+365+365+365 #2010 tp 2013 
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
        self.H1_day = saved_dict['H1_day']
        self.H2_day = saved_dict['H2_day']
        self.S = saved_dict['S'] 
        self.P = saved_dict['P'] 
        self.R = saved_dict['R'] 
        self.u = saved_dict['u'] 
        self.obs_data = saved_dict['obs_data'] 
        self.H = np.dot(self.S,self.Ae)
        
            
    def load_file(self,dir_path):
        id_path_unsorted = dir_path + '/rapid_data/rivid.csv'
        id_path_sorted = dir_path + '/rapid_data/riv_bas_id_San_Guad_hydroseq.csv'
        connectivity_path = dir_path + '/rapid_data/rapid_connect_San_Guad.csv'
        m3riv_path = dir_path + '/rapid_data/m3_riv.csv'
        m3riv_d_path = dir_path + '/rapid_data/m3_d_riv.csv'
        x_path = dir_path + '/rapid_data/x_San_Guad_2004_1.csv'
        k_path = dir_path + '/rapid_data/k_San_Guad_2004_1.csv'
        obs_path = dir_path + '/rapid_data/Qobs_San_Guad_2010_2013_full.csv'
        obs_id_path = dir_path + '/rapid_data/obs_tot_id_San_Guad_2010_2013_full.csv'
        vic_model_path = dir_path + '/rapid_data/m3_riv_vic_month.csv'
        ens_model_path = dir_path + '/rapid_data/m3_riv_ens_month.csv'
        
        params = {
                'id_path_unsorted': id_path_unsorted,
                'id_path_sorted': id_path_sorted,
                'connectivity_path': connectivity_path,
                'm3riv_path': m3riv_path,
                'm3riv_d_path':m3riv_d_path,
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
        
        # Introduction of the model parameters#
        # Corresponding to  Eq (6)&(7) in "Underlying fundamentals of kalman filtering for river network modeling"
        # A4 = [I-C1N]^-1(C3+C2N)] 
        # A5 = [I-C1N]^-1(C1+C2)] 
        # Ae += (96-p)/96 * A4^p @ A5 where p from 0 to 95
        # A0 += 1/96 * A4^p where p from 1 to 96
        # H1 += A4^p @ A5 where p from 0 to 95
        # H2 = A4^96
        # H1_day and H2_day are designed for delta_t = 1 day, while H1&H2 are 15mins
        # H1_day = A5 
        # H2_day = A4 
        
        self.Ae, self.A0, self.S, self.P, self.R, self.u, self.obs_data, \
            self.A4, self.A5, self.H1, self.H2, self.H1_day, self.H2_day= dataProcessor.pre_processing(**params)
        
        saved_dict = {
                'Ae': self.Ae,
                'A0': self.A0,
                'A4': self.A4,
                'A5': self.A5,
                'H1': self.H1,
                'H2': self.H2,
                'H1_day': self.H1_day,
                'H2_day': self.H2_day,
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

        ## Check the system observability (commented out for now)
        n = self.x.shape[0]
        O_mat = self.H
        for i in range(1, n):
            O_mat = np.vstack((O_mat, self.H))
            
        rank_O = np.linalg.matrix_rank(O_mat)
        
        if rank_O < n:
            print(f"rank of O: {rank_O} < n: {n}, system is not observable")
        else:
            print(f"rank of O: {rank_O} == n: {n}, system is observable")  
                  
        self.Q0 = np.zeros_like(self.u[0])
        print(f"rank of P:{np.linalg.matrix_rank(self.P)}, shape: {self.P.shape}")
        for timestep in range(self.days):
            if timestep <= 10000:
                x_predict = self.predict(self.u[timestep])
            else:
                x_predict = self.predict()
                
            # self.update(self.obs_data[timestep]) #comment it out for open loop
            self.update_discharge()
                
            kf_estimation.append(copy.deepcopy(self.getState())) 
            discharge_estimation.append(copy.deepcopy(self.getQ0()))
            open_loop_x.append(copy.deepcopy(self.getQ0()))

        np.savetxt("model_saved/discharge_est.csv", discharge_estimation, delimiter=",")
        np.savetxt("model_saved/river_lateral_est.csv", kf_estimation, delimiter=",")
        np.savetxt("model_saved/open_loop_est.csv", open_loop_x, delimiter=",")
        
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
        
        # self.H = self.S @ self.H1_day
        z = z - np.dot(self.S, np.dot(self.A0,self.Q0))
        z = z - np.dot(self.S, np.dot(self.H2_day,self.Q0))
        
        if inputType is not None:
            self.u, self.u_var = self.input_estimation(z)
            self.x = self.x + np.dot(self.B,self.u)
            innovation=  z - np.dot(self.H, self.x)
        else: 
            innovation = z - np.dot(self.H, self.x)
        
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        if np.linalg.matrix_rank(S) < S.shape[0]:
            delta_S = 0.00001*np.eye(S.shape[0])
            S += delta_S
            print(f"warning: S is not full rank")
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  
        self.x = self.x + np.dot(K, innovation)
        # self.P = self.P - np.dot(np.dot(K,self.H),self.P) 
    
    def update_discharge(self):
        self.Q0 = self.H1_day @ self.x + self.H2_day @ self.Q0
               
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
import itertools
import os
import pickle
import copy
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.stats import multivariate_normal as scipy_gaussian
from utility import PreProcessor
from typing import Optional, Dict, Any
from tqdm import tqdm
from main import greedy_schedule
class RAPIDKF:
    """
    Class for the Kalman Filter (KF) model used in river network modeling.
    
    Attributes:
        epsilon (float): Threshold for muskingum parameter.
        radius (int): Radius for KF estimation.
        i_factor (float): Scaling factor for covariance P.
        days (int): Total number of days from 2010 to 2013.
        month (int): Total months calculated from days.
        timestep (int): Current timestep in the simulation.
    """

    def __init__(self, load_mode: int = 0) -> None:
        """
        Initializes the RAPIDKF class.

        Args:
            load_mode (int): Mode for loading data (0 = file, 1 = pickle, 2 = both).
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.sub_dir_path = "model_saved_3hour"
        # Create directory if it doesn't exist
        if not os.path.exists(os.path.join(dir_path, "model_saved_3hour")):
            os.makedirs(os.path.join(dir_path, "model_saved_3hour"), exist_ok=True)
        self.epsilon: float = 0  # Muskingum parameter threshold
        self.radius: int = 10
        self.i_factor: float = 2.58  # Enforced on covariance P
        self.days: int = 366 + 365 + 365 + 365  # 2010 to 2013
        self.month: int = self.days // 365 * 12
        self.timestep: int = 0

        self.K = 2 #number of sensors to be selected
        
        if load_mode in [0, 2]:
            self.load_file(dir_path)
        if load_mode in [1, 2]:
            self.load_pkl(dir_path)

        self.V_full = np.identity(self.S.shape[0])
        self.S_g = []
        print("finished init")

    def load_pkl(self, dir_path: str) -> None:
        """
        Loads the saved model data from a pickle file.

        Args:
            dir_path (str): Directory path of the pickle file.
        """
        print("loading")
        dis_name = os.path.join(self.sub_dir_path,'load_coef.pkl')
        with open(os.path.join(dir_path, dis_name), 'rb') as f:
            saved_dict: Dict[str, Any] = pickle.load(f)
        print("finished loading")
        self.Ae = saved_dict['Ae']
        self.A0 = saved_dict['A0']
        self.Ae_day = saved_dict['Ae_day']
        self.A0_day = saved_dict['A0_day']
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

    def load_file(self, dir_path: str) -> None:
        """
        Loads model parameters from CSV files and initializes the model.

        Args:
            dir_path (str): Directory path where the CSV files are stored.
        """
        # Introduction to the model parameters:
        # These correspond to Equations (6) and (7) in the paper:
        # "Underlying fundamentals of Kalman filtering for river network modeling"

        # Matrix definitions:
        # A4 = [I - C1N]^-1(C3 + C2N)
        # A5 = [I - C1N]^-1(C1 + C2)
        # Ae is used for open-loop simulation, while Ae_day is for Kalman Filter (KF) estimation.
        # Parameter updates (open-loop and KF estimation):
        # Ae += (12 - p)/12 * A4^p @ A5  where p ranges from 0 to 11
        # A0 += 1/12 * A4^p  where p ranges from 1 to 12
        # Ae_day += (96 - p)/96 * A4^p @ A5  where p ranges from 0 to 95
        # A0_day += 1/96 * A4^p  where p ranges from 1 to 96
        # H1 and H2 updates:
        # H1 += A4^p @ A5  where p ranges from 0 to 95
        # H2 = A4^96
        # H1_day and H2_day parameters are designed for a daily timestep (delta_t = 1 day),
        # while H1 and H2 are designed for a 15-minute timestep.
        # H1_day = A5
        # H2_day = A4

        params = {
            'id_path_unsorted': os.path.join(dir_path, 'rapid_data/rivid.csv'),
            'id_path_sorted': os.path.join(dir_path, 'rapid_data/riv_bas_id_San_Guad_hydroseq.csv'),
            'connectivity_path': os.path.join(dir_path, 'rapid_data/rapid_connect_San_Guad.csv'),
            'm3riv_path': os.path.join(dir_path, 'rapid_data/m3_riv.csv'),
            # 'm3riv_d_path': os.path.join(dir_path, 'rapid_data/m3_d_riv.csv'),
            'x_path': os.path.join(dir_path, 'rapid_data/x_San_Guad_2004_1.csv'),
            'k_path': os.path.join(dir_path, 'rapid_data/k_San_Guad_2004_1.csv'),
            'obs_path': os.path.join(dir_path, 'rapid_data/Qobs_San_Guad_2010_2013_full.csv'),
            'obs_id_path': os.path.join(dir_path, 'rapid_data/obs_tot_id_San_Guad_2010_2013_full.csv'),
            'vic_model_path': os.path.join(dir_path, 'rapid_data/m3_riv_vic_month.csv'),
            'ens_model_path': os.path.join(dir_path, 'rapid_data/m3_riv_ens_month.csv'),
            'days': self.days,
            'month': self.month,
            'radius': self.radius,
            'epsilon': self.epsilon,
            'i_factor': self.i_factor,
        }

        data_processor = PreProcessor()
        self.Ae, self.A0, self.Ae_day, self.A0_day, self.S, self.P, self.R, self.u, self.obs_data, \
            self.A4, self.A5, self.H1, self.H2, self.H1_day, self.H2_day = data_processor.pre_processing(**params)

        saved_dict = {
            'Ae': self.Ae,
            'A0': self.A0,
            'Ae_day': self.Ae_day,
            'A0_day': self.A0_day,
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

        dis_name = os.path.join(self.sub_dir_path,'load_coef.pkl')
        with open(os.path.join(dir_path, dis_name), 'wb') as f:
            pickle.dump(saved_dict, f)

    def simulate(self, sim_mode: int = 0) -> None:
        """
        Simulates the Kalman Filter model.

        Args:
            sim_mode (int): Mode for simulation (0 = open loop, 1 = Kalman Filter estimation).
        """
        kf_estimation = []
        discharge_estimation = []
        open_loop_x = []

        self.H = np.dot(self.S, self.Ae_day)
        self.Q0 = np.zeros_like(self.u[0])
            
        # ## Check the system observability (commented out for now)
        # n = self.x.shape[0]
        # O_mat = self.H 
        # O_mat = self.S 
        
        # for i in range(1, n):
        #     # O_mat = np.vstack((O_mat, self.H))
        #     O_mat = np.vstack((O_mat, self.S @ np.linalg.matrix_power(self.Ae, i)))
            
        # rank_O = np.linalg.matrix_rank(O_mat)
        
        # if rank_O < n:
        #     print(f"rank of O: {rank_O} < n: {n}, system is not observable")
        # else:
        #     print(f"rank of O: {rank_O} == n: {n}, system is observable")  
        for timestep in tqdm(range(self.days)):
            discharge_avg = np.zeros_like(self.u[0])
            self.x = np.zeros_like(self.u[0])
            evolution_steps = 8  # Number of steps for each day

            if sim_mode == 1:
                
                sigma_greedy, s_greedy = greedy_schedule(
                    A=self.Ae_day,
                    C=self.H,
                    W=np.identity(self.Ae_day.shape[0]),
                    V=self.V_full,
                    S0=self.P,
                    K=self.K,
                    l=1,  # Schedule for the current timestep only
                )
                selected_sensors = s_greedy[0]
                with open("sigma.txt",'a') as f:
                    f.write(f"{sigma_greedy}\n")
                self.S_g.append(selected_sensors)

                # Kalman Filter estimation (updates every 3 hours)
                for i in range(evolution_steps):
                    self.x += self.u[timestep * evolution_steps + i] / evolution_steps

                self.update(self.obs_data[timestep], timestep, selected_sensors)

                for i in range(evolution_steps):
                    discharge_avg += self.update_discharge()

            elif sim_mode == 0:
                # Open-loop simulation (predict inflow every 3 hours)
                for i in range(evolution_steps):
                    self.predict(self.u[timestep * evolution_steps + i])
                    discharge_avg += self.update_discharge()

            discharge_avg /= evolution_steps

            kf_estimation.append(copy.deepcopy(self.get_state()))
            discharge_estimation.append(discharge_avg)
            open_loop_x.append(copy.deepcopy(self.get_discharge()))

        # Save results to the created directory
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = os.path.join(dir_path,self.sub_dir_path)
        np.savetxt(os.path.join(dir_path, "discharge_est.csv"), discharge_estimation, delimiter=",")
        np.savetxt(os.path.join(dir_path, "river_lateral_est.csv"), kf_estimation, delimiter=",")
        np.savetxt(os.path.join(dir_path, "open_loop_est.csv"), open_loop_x, delimiter=",")
        
    def predict(self, u: Optional[np.ndarray] = None) -> None:
        """
        Predicts the next state of the system.

        Args:
            u (np.ndarray): Optional inflow data for the prediction.
        """
        self.x = u if u is not None else np.zeros_like(self.x)
        # self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        
        self.timestep += 1

    def update(self, z: np.ndarray, timestep: int, input_type: Optional[str] = None) -> None:
        """
        Updates the Kalman Filter with new measurements.

        Args:
            z (np.ndarray): Observation data.
            timestep (int): Current timestep.
            input_type (str): Input type (unused in this model).
        """
        diag_R = 0.01 * z ** 2
        self.R = np.diag(diag_R)
        z = z - np.dot(self.S, np.dot(self.A0_day, self.Q0))
        innovation = z - np.dot(self.H, self.x)

        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x += np.dot(K, innovation)
        # self.P = self.P - np.dot(np.dot(K,self.H),self.P) 

    def update(self, z: np.ndarray, timestep: int, S: tuple) -> None:
        """
        Updates the Kalman Filter with new measurements from selected sensors.
        
        Args:
            z (np.ndarray): Observation data.
            timestep (int): Current timestep.
            S (tuple): Selected sensor indices (1-based).
        """
        # Convert 1-based sensor indices to 0-based
        S_zero_based = [sensor - 1 for sensor in S]
        
        # Select measurements from the selected sensors
        z_selected = z[S_zero_based]
        
        # Build the measurement matrix for selected sensors
        H_selected = self.H[S_zero_based, :]
        
        # Build the measurement noise covariance for selected sensors
        V_selected = self.V_full[np.ix_(S_zero_based, S_zero_based)]
        
        # Compute the Kalman Gain
        innovation = z_selected - H_selected @ self.x
        S_matrix = V_selected + H_selected @ self.P @ H_selected.T
        K = self.P @ H_selected.T @ np.linalg.inv(S_matrix)
        
        # Update the state estimate and covariance
        self.x += K @ innovation
        self.P = (np.eye(self.P.shape[0]) - K @ H_selected) @ self.P

    def update_discharge(self) -> np.ndarray:
        """
        Updates the discharge using the current state.

        Returns:
            np.ndarray: Averaged discharge.
        """
        ### Method1
        Q0_ave = np.zeros_like(self.Q0)
        for _ in range(12):
            self.Q0 = self.A5 @ self.x + self.A4 @ self.Q0
            Q0_ave += self.Q0
            
        # ### Method2
        # self.Q0 = self.H1 @ self.x + self.H2 @ self.Q0
        # Q0_ave = self.Q0

        return Q0_ave / 12

    def get_state(self) -> np.ndarray:
        """
        Returns the current state.

        Returns:
            np.ndarray: Current state.
        """
        return self.x

    def get_discharge(self) -> np.ndarray:
        """
        Returns the current discharge.

        Returns:
            np.ndarray: Current discharge.
        """
        return self.Q0


if __name__ == '__main__':
    rapid_kf = RAPIDKF(load_mode=1)
    rapid_kf.simulate(sim_mode=1)

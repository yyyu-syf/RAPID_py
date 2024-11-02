import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg


class PreProcessor:
    """
    A class responsible for preprocessing the data required by the Kalman Filter model.
    """
    
    def __init__(self) -> None:
        """
        Initializes the PreProcessor class.
        """
        pass

    def pre_processing(self, **kwargs):
        """
        Processes input data and sets up various matrices for the Kalman Filter.

        Args:
            **kwargs: Dictionary containing paths and configuration parameters.
        """
        # Extracting parameters from the provided dictionary
        id_path_unsorted = kwargs['id_path_unsorted']
        id_path_sorted = kwargs['id_path_sorted']
        connectivity_path = kwargs['connectivity_path']
        m3riv_path = kwargs['m3riv_path']
        # m3riv_d_path = kwargs['m3riv_d_path']
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

        # Load data from CSV files
        reach_id_unsorted = pd.read_csv(id_path_unsorted, header=None).to_numpy().flatten()
        reach_id_sorted = pd.read_csv(id_path_sorted, header=None).to_numpy().flatten()
        connect_data = pd.read_csv(connectivity_path, header=None)
        m3riv_data = pd.read_csv(m3riv_path, header=None)
        # m3riv_d_data = pd.read_csv(m3riv_d_path, header=None)
        x_data = pd.read_csv(x_path, header=None)
        k_data = pd.read_csv(k_path, header=None)
        obs_data = pd.read_csv(obs_path, header=None)
        obs_id = pd.read_csv(obs_id_path, header=None)
        vic_data_m = pd.read_csv(vic_model_path, header=None)
        ens_data_m = pd.read_csv(ens_model_path, header=None)

        # Cutoff data based on preference
        cutoff = len(reach_id_sorted)
        reach_id_unsorted = reach_id_unsorted[:cutoff]
        connect_data = connect_data[:cutoff]
        x_data = x_data[:cutoff]
        k_data = k_data[:cutoff]
        vic_data_m = vic_data_m.iloc[1:self.month+1, :cutoff]
        ens_data_m = ens_data_m.iloc[1:self.month+1, :cutoff]
        m3riv_data = m3riv_data.iloc[1:self.days*8+1, :cutoff]
        # m3riv_d_data = m3riv_d_data.iloc[1:self.days+1, :cutoff]
        obs_data = obs_data.iloc[:self.days]
        self.l_reach = x_data.shape[0]

        print(f"Number of reaches: {self.l_reach}")
        print(f"3-hourly m3riv_data shape: {m3riv_data.shape}")
        # print(f"Daily m3riv_data shape: {m3riv_d_data.shape}")
        print(f"Number of observations: {obs_id.shape}")

        # Process lateral inflow
        lateral_daily = m3riv_data.to_numpy()
        lateral_daily_averaged = lateral_daily / 3 / 3600
        lateral_daily_averaged_sorted = np.zeros_like(lateral_daily_averaged)

        # Sorting lateral daily data based on reach IDs
        for id_reach in reach_id_unsorted:
            idx = np.where(reach_id_unsorted == id_reach)[0]
            sorted_idx = np.where(reach_id_sorted == id_reach)[0]
            lateral_daily_averaged_sorted[:, sorted_idx] = lateral_daily_averaged[:, idx]

        np.savetxt("model_saved_3hour/u.csv", lateral_daily_averaged_sorted, delimiter=",")
        print(f"Lateral daily data shape: {lateral_daily_averaged_sorted.shape}")

        # Process Muskingum parameters
        self._process_muskingum_params(x_data, k_data, reach_id_unsorted, reach_id_sorted)

        # Calculate connectivity matrix
        self.calculate_connectivity(connect_data, reach_id_sorted)

        # Calculate dynamics coefficients
        self.calculate_coefficient()

        # Set up observation matrix S
        obs_id = obs_id.to_numpy().flatten()
        S = np.zeros((len(obs_id), len(reach_id_sorted)), dtype=int)
        for i, obs in enumerate(obs_id):
            index = np.where(reach_id_sorted == obs)[0]
            S[i, index] = 1

        # Initial state for matrix R
        R0 = 0.1 * obs_data.to_numpy()[0]
        R = np.diag(R0 ** 2)
        print(f"Dim of R: {R.shape}")

        # Calculate and prune P based on radius
        delta = abs((vic_data_m - ens_data_m)).sum(axis=0) / self.days / 24 / 3600
        P = self.i_factor * np.dot(delta.values.reshape(-1, 1), delta.values.reshape(-1, 1).T)
        pruned_P = self.pruneP(P, connect_data, self.radius, reach_id_sorted)
        print(f"Dim of P: {P.shape}")

        # Save processed data
        self._save_matrices(S, R, P, pruned_P, lateral_daily_averaged_sorted)

        return self.Ae, self.A0, self.Ae_day, self.A0_day, S, pruned_P, R, lateral_daily_averaged_sorted, \
               obs_data.to_numpy(), self.A4, self.A5, self.H1, self.H2, self.H1_day, self.H2_day

    def _process_muskingum_params(self, x_data, k_data, reach_id_unsorted, reach_id_sorted):
        """
        Process Muskingum parameters from data and calculate coefficients.
        """
        self.musking_k = np.zeros(self.l_reach)
        self.musking_x = np.zeros(self.l_reach)
        self.musking_C1 = np.zeros(self.l_reach)
        self.musking_C2 = np.zeros(self.l_reach)
        self.musking_C3 = np.zeros(self.l_reach)
        self.musking_C1_day = np.zeros(self.l_reach)
        self.musking_C2_day = np.zeros(self.l_reach)
        self.musking_C3_day = np.zeros(self.l_reach)

        # Handle uniform x_data case
        if x_data.nunique().iloc[0] == 1:
            self.musking_x = np.full(self.l_reach, x_data.iloc[0, 0])
            print(f"All Muskingum x values are the same: {x_data.iloc[0, 0]}")
        else:
            for i, x in enumerate(x_data):
                idx = reach_id_unsorted[i]
                sorted_idx = np.where(reach_id_sorted == idx)[0]
                self.musking_x[sorted_idx] = x

        for i, k in enumerate(k_data.values.reshape(-1)):
            idx = reach_id_unsorted[i]
            sorted_idx = np.where(reach_id_sorted == idx)[0]
            self.musking_k[sorted_idx] = k  # Unit: seconds

        # Calculate Muskingum coefficients
        for i in range(len(k_data)):
            k = self.musking_k[i]
            x = self.musking_x[i]
            delta_t = 15 * 60  # 15 minutes
            self.musking_C1[i], self.musking_C2[i], self.musking_C3[i] = self.calculate_Cs(k, x, delta_t)

            # For daily time step
            delta_t_day = 24 * 60 * 60
            self.musking_C1_day[i], self.musking_C2_day[i], self.musking_C3_day[i] = self.calculate_Cs(k, x, delta_t_day)

        # Convert Muskingum parameters to diagonal matrices
        self.musking_C1 = np.diag(self.musking_C1)
        self.musking_C2 = np.diag(self.musking_C2)
        self.musking_C3 = np.diag(self.musking_C3)
        self.musking_C1_day = np.diag(self.musking_C1_day)
        self.musking_C2_day = np.diag(self.musking_C2_day)
        self.musking_C3_day = np.diag(self.musking_C3_day)

    def calculate_connectivity(self, river_network, reach_id_sorted):
        """
        Calculates the connectivity matrix based on river network data.
        """
        column_names = ['id', 'downId', 'numUp', 'upId1', 'upId2', 'upId3', 'upId4']
        river_network.columns = column_names

        connectivity_matrix = np.zeros((self.l_reach, self.l_reach), dtype=int)

        for _, row in river_network.iterrows():
            cur_id = row[0]
            up_row_index = np.where(reach_id_sorted == cur_id)[0]
            downstream_id = row[1]
            row_index = np.where(reach_id_sorted == downstream_id)[0]

            if row_index < self.l_reach and up_row_index < self.l_reach:
                connectivity_matrix[row_index, up_row_index] = 1

        self.N = connectivity_matrix

        # Save connectivity plot
        w = connectivity_matrix.shape[0]
        plt.figure(figsize=(8, 8))
        plt.imshow(connectivity_matrix, cmap='Greys', interpolation='none')
        plt.title("Connectivity Density")
        plt.xticks(ticks=np.arange(0, w, w/10), labels=np.arange(0, w, w/10))
        plt.yticks(ticks=np.arange(0, w, w/10), labels=np.arange(0, w, w/10))
        plt.grid(color='gray', linestyle='-', linewidth=0.5)
        plt.savefig("model_saved_3hour/connectivity.png", dpi=300, bbox_inches='tight')

    def calculate_coefficient(self):
        """
        Calculates the dynamic coefficients Ae, A0, and related matrices.
        """
        # Calculate 3hour-based dynamics coefficients
        mat_I = np.identity(self.l_reach)
        A1 = mat_I - self.musking_C1 @ self.N
        A1_inv = np.linalg.inv(A1)
        A1_inv[abs(A1_inv) < self.epsilon] = 0  # Filter small values

        A2 = self.musking_C1 + self.musking_C2
        A3 = self.musking_C3 + self.musking_C2 @ self.N
        A4 = A1_inv @ A3
        A5 = A1_inv @ A2

        n_12 = 12  # 3-hourly = 12 * 15mins
        Ae = np.zeros((self.l_reach, self.l_reach))
        for p in np.arange(0, n_12):
            Ae += (n_12 - p) / n_12 * np.linalg.matrix_power(A4, p)
        Ae = Ae @ A5

        A0 = np.zeros((self.l_reach, self.l_reach))
        for p in np.arange(1, n_12 + 1):
            A0 += 1 / n_12 * np.linalg.matrix_power(A4, p)

        self.H1 = np.zeros_like(Ae)
        for p in np.arange(0, n_12):
            self.H1 += np.linalg.matrix_power(A4, p)
        self.H1 = self.H1 @ A5
        self.H2 = np.linalg.matrix_power(A4, n_12)

        self.Ae = Ae
        self.A0 = A0
        self.A4 = A4
        self.A5 = A5

        # Calculate day-based dynamics coefficients
        mat_I_day = np.identity(self.l_reach)
        A1_day = mat_I_day - np.dot(self.musking_C1_day, self.N)
        A1_inv_day = np.linalg.inv(A1_day)
        A1_inv_day[A1_inv_day < self.epsilon] = 0
        A2_day = self.musking_C1_day + self.musking_C2_day
        A3_day = self.musking_C3_day + np.dot(self.musking_C2_day, self.N)
        A4_day = A1_inv_day @ A3_day
        A5_day = A1_inv_day @ A2_day

        n_96 = 96  # 1 day = 96 * 15mins
        Ae_day = np.zeros((self.l_reach, self.l_reach))
        for p in np.arange(0, n_96):
            Ae_day += (n_96 - p) / n_96 * np.linalg.matrix_power(A4_day, p)
        Ae_day = Ae_day @ A5_day

        A0_day = np.zeros((self.l_reach, self.l_reach))
        for p in np.arange(1, n_96 + 1):
            A0_day += 1 / n_96 * np.linalg.matrix_power(A4_day, p)

        self.H1_day = A5_day
        self.H2_day = A4_day
        self.A0_day = A0_day
        self.Ae_day = Ae_day

    def calculate_Cs(self, k, x, delta_t):
        """
        Calculate the Muskingum coefficients C1, C2, and C3.

        Args:
            k (float): The Muskingum k parameter.
            x (float): The Muskingum x parameter.
            delta_t (int): The time step (seconds).

        Returns:
            Tuple[float, float, float]: C1, C2, and C3 values.
        """
        C1 = (delta_t / 2 - k * x) / (k * (1 - x) + delta_t / 2)
        C2 = (delta_t / 2 + k * x) / (k * (1 - x) + delta_t / 2)
        C3 = (k * (1 - x) - delta_t / 2) / (k * (1 - x) + delta_t / 2)
        return C1, C2, C3

    def pruneP(self, P, river_network, radius, reach_id_sorted):
        """
        Prunes the matrix P based on the river network and a specified radius.

        Args:
            P (np.ndarray): The matrix to prune.
            river_network (pd.DataFrame): The river network data.
            radius (int): The radius for pruning.
            reach_id_sorted (np.ndarray): Sorted reach IDs.

        Returns:
            np.ndarray: Pruned matrix P.
        """
        column_names = ['id', 'downId', 'numUp', 'upId1', 'upId2', 'upId3', 'upId4']
        river_network.columns = column_names
        maskP = np.zeros_like(P)

        for _, row in river_network.iterrows():
            cur_id = row[0]
            row_index = np.where(reach_id_sorted == cur_id)[0]
            maskP[row_index, row_index] = 1
            downStreamId = row[0]

            # Downstream search
            for _ in range(radius):
                condition = river_network['id'] == downStreamId
                condition2 = reach_id_sorted == downStreamId
                if condition2.any():
                    down_row_index = river_network.index[condition].tolist()
                    down_row_index2 = np.where(condition2)[0]

                if row_index < self.l_reach and down_row_index2 < self.l_reach:
                    maskP[row_index, down_row_index2] = 1
                    maskP[down_row_index2, row_index] = 1
                    downStreamId = river_network.iloc[down_row_index]['downId'].tolist()[0]

            # Upstream search
            upstream_queue = [(cur_id, 0)]
            visited = set()
            while upstream_queue:
                upStreamId, depth = upstream_queue.pop(0)
                if depth >= radius:
                    continue

                condition = river_network['id'] == upStreamId
                condition2 = reach_id_sorted == upStreamId
                if condition2.any():
                    up_row_index2 = np.where(condition2)[0]

                    if row_index < self.l_reach and up_row_index2 < self.l_reach:
                        maskP[row_index, up_row_index2] = 1
                        maskP[up_row_index2, row_index] = 1

                        if upStreamId not in visited:
                            visited.add(upStreamId)
                            numUp = river_network.loc[condition, 'numUp'].values[0]
                            for i in range(1, numUp + 1):
                                next_upStreamId = river_network.loc[condition, f'upId{i}'].values[0]
                                if pd.notna(next_upStreamId):
                                    upstream_queue.append((next_upStreamId, depth + 1))

        # Save mask plot
        w = P.shape[0]
        plt.figure(figsize=(8, 8))
        plt.imshow(maskP, cmap='Greys', interpolation='none')
        plt.title(f"P density with R = {radius}")
        plt.xticks(ticks=np.arange(0, w, w / 10), labels=np.arange(0, w, w / 10))
        plt.yticks(ticks=np.arange(0, w, w / 10), labels=np.arange(0, w, w / 10))
        plt.grid(color='gray', linestyle='-', linewidth=0.5)
        plt.savefig("model_saved_3hour/density_P.png", dpi=300, bbox_inches='tight')

        return P * maskP

    def _save_matrices(self, S, R, P, pruned_P, lateral_daily_averaged_sorted):
        """
        Save the matrices to CSV files.

        Args:
            S (np.ndarray): Observation matrix.
            R (np.ndarray): Measurement noise covariance matrix.
            P (np.ndarray): Covariance matrix.
            pruned_P (np.ndarray): Pruned covariance matrix.
            lateral_daily_averaged_sorted (np.ndarray): Sorted lateral daily data.
        """
        np.savetxt("model_saved_3hour/k.csv", self.musking_k, delimiter=",")
        np.savetxt("model_saved_3hour/P_delta.csv", P, delimiter=",")
        np.savetxt("model_saved_3hour/P.csv", P, delimiter=",")
        np.savetxt("model_saved_3hour/prunedP.csv", pruned_P, delimiter=",")
        np.savetxt("model_saved_3hour/S.csv", S, delimiter=",")
        np.savetxt("model_saved_3hour/R.csv", R, delimiter=",")
        np.savetxt("model_saved_3hour/z.csv", S, delimiter=",")
        np.savetxt("model_saved_3hour/u.csv", lateral_daily_averaged_sorted, delimiter=",")
        np.savetxt("model_saved_3hour/Ae.csv", self.Ae, delimiter=",")
        np.savetxt("model_saved_3hour/A0.csv", self.A0, delimiter=",")
        np.savetxt("model_saved_3hour/A4.csv", self.A4, delimiter=",")
        np.savetxt("model_saved_3hour/A5.csv", self.A5, delimiter=",")
        np.savetxt("model_saved_3hour/N.csv", self.N, delimiter=",")

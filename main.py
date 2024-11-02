import numpy as np
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
def build_nonzeroC(C, S):
    """
    Extracts the measurement matrices for the selected sensors.

    Parameters:
    - C (ndarray): Measurement matrices for all sensors, shape (G, n).
    - S (tuple or list): Current set of selected sensor indices (1-based).

    Returns:
    - C_hat (ndarray): Measurement matrices for selected sensors, shape (K, n).
    """
    S_sorted = sorted(S)
    C_hat = C[np.array(S_sorted) - 1, :]  # Convert to 0-based indexing
    return C_hat

def build_nonzeroV(V, S):
    """
    Extracts the measurement noise covariance submatrix for selected sensors.

    Parameters:
    - V (ndarray): Measurement noise covariance matrix, shape (G, G).
    - S (tuple or list): Current set of selected sensor indices (1-based).

    Returns:
    - V_hat (ndarray): Measurement noise covariance for selected sensors, shape (K, K).
    """
    S_sorted = sorted(S)
    V_hat = V[np.ix_(np.array(S_sorted) - 1, np.array(S_sorted) - 1)]  # Convert to 0-based indexing
    return V_hat

def cov_matrix(S, Sigma_prev, A, W, V, C):
    """
    Computes the updated covariance matrix given the selected sensors.

    Parameters:
    - S (tuple or list): Current set of selected sensor indices (1-based).
    - Sigma_prev (ndarray): Previous covariance matrix, shape (n, n).
    - A (ndarray): State transition matrix, shape (n, n).
    - W (ndarray): Process noise covariance matrix, shape (n, n).
    - V (ndarray): Measurement noise covariance matrix, shape (G, G).
    - C (ndarray): Measurement matrices for all sensors, shape (G, n).

    Returns:
    - Sigma (ndarray): Updated covariance matrix, shape (n, n).
    """
    K = len(S)

    if K == V.shape[0]:
        V_hat = V
        C_hat = C
    else:
        V_hat = build_nonzeroV(V, S)
        C_hat = build_nonzeroC(C, S)

    # Compute A * Sigma_prev * A^T + W
    ASigmaA_T = A @ Sigma_prev @ A.T + W

    # Compute the inverse of (A * Sigma_prev * A^T + W)
    try:
        inv_ASigmaA_T = np.linalg.inv(ASigmaA_T)
    except np.linalg.LinAlgError:
        # If singular, use pseudo-inverse
        inv_ASigmaA_T = np.linalg.pinv(ASigmaA_T)

    # Compute C_hat^T * V_hat * C_hat
    CVC = C_hat.T @ V_hat @ C_hat

    # Compute the inverse of (inv_ASigmaA_T + CVC)
    try:
        Sigma = np.linalg.inv(inv_ASigmaA_T + CVC)
    except np.linalg.LinAlgError:
        # If singular, use pseudo-inverse
        Sigma = np.linalg.pinv(inv_ASigmaA_T + CVC)

    return Sigma

def greedy_schedule(A, C, W, V, S0, K, l):
    """
    Greedy_Schedule selects a sequence of sensor sets over l time steps to minimize 
    the trace of the covariance matrix using a greedy approach.

    Parameters:
    - A (ndarray): State transition matrix, shape (n, n).
    - C (ndarray): Measurement matrices for all sensors, shape (G, n).
    - W (ndarray): Process noise covariance matrix, shape (n, n).
    - V (ndarray): Measurement noise covariance matrix, shape (G, G).
    - S0 (ndarray): Initial covariance matrix, shape (n, n).
    - K (int): Number of sensors to select at each time step.
    - l (int): Number of time steps.

    Returns:
    - Sigma_g (ndarray): Covariance matrices over time, shape (n, n, l+1).
    - S_g (list of tuples): Selected sensor sets for each time step.
    """
    n = A.shape[0]
    G = C.shape[0]
    Q = list(range(1, G + 1))  # 1-based sensor indices
    S_all = list(itertools.combinations(Q, K))
    Sigma_g = np.zeros((n, n, l + 1))
    Sigma_g[:, :, 0] = S0
    S_g = [None] * l

    for t in range(1, l + 1):
        # Predict the next state covariance before selecting sensors
        Sigma_pred = A @ Sigma_g[:, :, t - 1] @ A.T + W
        prev_trace = np.trace(Sigma_pred)
        print("prev_trace",prev_trace)
        obj_cur_max = float('-inf')
        currentBestSet = None
        currentBestSigma = None

        for S_cur in tqdm(S_all):
            Sigma_temp = cov_matrix(S_cur, Sigma_g[:, :, t - 1], A, W, V, C)
            cur_trace = np.trace(Sigma_temp)

            marginal_gain = prev_trace - cur_trace

            if abs(marginal_gain) > obj_cur_max:
                obj_cur_max = marginal_gain
                currentBestSet = S_cur
                currentBestSigma = Sigma_temp

        S_g[t - 1] = currentBestSet
        Sigma_g[:, :, t] = currentBestSigma
        print("sigma_g",Sigma_g)
    return Sigma_g, S_g

def brute_force_schedule(A, C, W, V, S0, K, l):
    """
    BruteForce_Schedule performs exhaustive search to find the optimal sensor scheduling
    over l time steps that minimizes the cumulative trace of covariance matrices.

    Parameters:
    - A (ndarray): State transition matrix, shape (n, n).
    - C (ndarray): Measurement matrices for all sensors, shape (G, n).
    - W (ndarray): Process noise covariance matrix, shape (n, n).
    - V (ndarray): Measurement noise covariance matrix, shape (G, G).
    - S0 (ndarray): Initial covariance matrix, shape (n, n).
    - K (int): Number of sensors to select at each time step.
    - l (int): Number of time steps.

    Returns:
    - Sigma_opt (list of ndarrays): Optimal covariance matrices for each time step, length l.
    - obj_maxes (list of floats): Maximum objective values for each time step, length l.
    """
    n = A.shape[0]
    G = C.shape[0]
    Q = list(range(1, G + 1))  # 1-based sensor indices

    # Generate all possible combinations of K sensors
    All_Combinations_each_time = list(itertools.combinations(Q, K))
    num_cases_each_time = len(All_Combinations_each_time)

    # Initialize Sigma_set as a list of l+1 elements (time steps 0 to l)
    Sigma_set = [None] * (l + 1)

    # Initialize Sigma_set[0] with all possible initial combinations set to S0
    Sigma_set[0] = np.zeros((n, n, num_cases_each_time))
    for i in range(num_cases_each_time):
        Sigma_set[0][:, :, i] = S0

    # Preallocate Sigma_set for time steps 1 to l
    for t in range(1, l + 1):
        num_matrices = num_cases_each_time ** t
        Sigma_set[t] = np.zeros((n, n, num_matrices))

    # Initialize outputs
    Sigma_opt = [None] * l
    obj_maxes = [float('-inf')] * l

    # Iterate over each time step
    for t in range(1, l + 1):
        obj_cur_max = float('-inf')
        currentBestSet = None
        currentBestSigma = None

        # Iterate over all possible sensor combinations
        for i, S_cur in enumerate(All_Combinations_each_time):
            # Iterate over all previous covariance matrices
            for j in range(num_cases_each_time ** (t - 1)):
                Sigma_prev = Sigma_set[t - 1][:, :, j]

                # Compute the updated covariance matrix
                Sigma_temp = cov_matrix(S_cur, Sigma_prev, A, W, V, C)

                # Calculate the new index for Sigma_set[t]
                index = i * (num_cases_each_time ** (t - 1)) + j

                # Assign the computed covariance matrix to Sigma_set[t]
                Sigma_set[t][:, :, index] = Sigma_temp

                # Compute the objective value (negative trace)
                obj_cur = -np.trace(Sigma_temp)

                # Update the best objective and corresponding sensor set if necessary
                if obj_cur > obj_cur_max:
                    obj_cur_max = obj_cur
                    currentBestSet = S_cur
                    currentBestSigma = Sigma_temp

        # Store the best covariance matrix and objective value for this time step
        Sigma_opt[t - 1] = currentBestSigma
        obj_maxes[t - 1] = obj_cur_max

        print(f"Brute-Force Scheduling - Time Step {t}: Best Objective = {obj_cur_max}, Best Sensor Set = {currentBestSet}")

    return Sigma_opt, obj_maxes

def main():
    # Initialize hyper-parameters
    N = 1        # Number of system cases
    l = 5        # Finite time horizon
    n = 2        # Dimension of the state
    G = 5        # Number of sensors in the ground set
    K = 3        # Number of sensors to select at each time step
    Q = list(range(1, G + 1))  # Sensor indices (1-based)
    W = np.eye(n) * 0.4        # Process noise covariance
    V = np.eye(G) * 0.7        # Measurement noise covariance
    x0 = np.zeros((n, 1))      # Initial state (unused in this script)
    S0 = np.eye(n)              # Initial covariance matrix

    # Accumulate traces over all simulations
    acc_tr_rand = np.zeros(l)
    acc_tr_g = np.zeros(l)
    acc_tr_opt = np.zeros(l)

    for i in range(1, N + 1):
        # Generate random measurement matrices C and state transition matrix A
        C = np.random.rand(G, n)
        A = np.random.rand(n, n)
        e = max(np.linalg.eigvals(A).real)
        while e >= 1:
            A = np.random.rand(n, n)
            e = max(np.linalg.eigvals(A).real)
            print(e)
        
        # Reset accumulated traces for this simulation
        acc_tr_rand = np.zeros(l)
        acc_tr_g = np.zeros(l)
        acc_tr_opt = np.zeros(l)

        # ---- Random Scheduling ----
        S_all = list(itertools.combinations(Q, K))
        num_combinations = len(S_all)
        S_rand = [S_all[np.random.randint(0, num_combinations)] for _ in range(l)]
        
        Sigma_rand = np.zeros((n, n, l + 1))
        Sigma_rand[:, :, 0] = S0

        # Compute covariance matrices for random scheduling
        for t in range(1, l + 1):
            S_t = S_rand[t - 1]
            Sigma_rand[:, :, t] = cov_matrix(S_t, Sigma_rand[:, :, t - 1], A, W, V, C)
        
        # ---- Greedy Scheduling ----
        Sigma_g, S_g = greedy_schedule(A, C, W, V, S0, K, l)
        
        # ---- Brute-Force Scheduling ----
        Sigma_opt, obj_maxes = brute_force_schedule(A, C, W, V, S0, K, l)
        
        # ---- Calculate Traces ----
        for t in range(l):
            acc_tr_rand[t] += np.trace(Sigma_rand[:, :, t + 1])
            acc_tr_g[t] += np.trace(Sigma_g[:, :, t + 1])
            acc_tr_opt[t] += np.trace(Sigma_opt[t])
        
        # ---- Plotting ----
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, l + 1), acc_tr_rand, '-k', linewidth=2, markersize=5, label='Random')
        plt.plot(range(1, l + 1), acc_tr_g, '-or', linewidth=2, markersize=5, markerfacecolor='r', label='Greedy')
        plt.plot(range(1, l + 1), acc_tr_opt, '-*g', linewidth=2, markersize=5, markerfacecolor='g', label='Brute Force')
        
        plt.xlabel('Time step')
        plt.ylabel('trace(Pₜ)')
        plt.title('Comparison of Optimization Methods')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Optionally, save the plot
        # plt.savefig(f'ComparisonPlot_Simulation_{i}.png')



if __name__ == "__main__":
    main()

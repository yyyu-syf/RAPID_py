# -*- coding: utf-8 -*-
'''
This KF is information format of kalman filter. 
'''
from scipy.stats import multivariate_normal as scipy_guassian
import numpy as np
import time

class InfoKalmanFilter(object):
    def __init__(self, A = None, B = None, H = None, S=None, Q = None, R = None, P = None, x0 = None):

        if(A is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = A.shape[0]
        self.m = H.shape[0]
        self.A = A
        self.H = H
        self.S = S
        self.B = np.eye(self.n) if B is None else B
        self.Q = np.eye(self.n)*0 if Q is None else Q 
        self.R = np.eye(self.m) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.u = np.zeros((self.B.shape[-1]))
        self.infoP = np.linalg.inv(self.P)
        self.x = np.zeros((self.n, 1)) if x0 is None else x0
        self.x_last = np.zeros((self.n, 1)) if self.x is None else self.x
        self.Q0 = np.zeros_like(self.u)
        self.timestep = 0
        self.W = np.zeros((self.n, self.m))
    
    def predict(self,u=None):
        if u is not None:
            # u = np.zeros((self.B.shape[-1], 1))
            self.x = u # only applicable for RAPID model

        # self.x = np.dot(self.A,self.x)
        # self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        self.timestep += 1
        
        return self.x


    # def input_estimation(self,z): 
    #     F = np.dot(self.H, self.B)
    #     S = np.dot(np.dot(self.H,self.P),self.H.T)+self.R
    #     M_1 = np.linalg.inv(np.dot(np.dot(F.T,np.linalg.inv(S)),F))
    #     M_2 = np.dot(F.T,np.linalg.inv(S))
    #     M = np.dot(M_1,M_2)
    #     innovation = z - np.dot(self.H, self.x)
    #     u = np.dot(M,innovation)
    #     u_var = M_1

    #     return u, u_var 

    def updateInfoKF(self, z, inputType=None): 
        '''
        Update based on information filter, for decentralized purpose
        '''
        ### In RAPID model, inputType = None
        diag_R = 0.01*z**2
        self.R = np.diag(diag_R)
        dI = np.eye(self.R.shape[0]) * 0.0001 #avoid singularity 
        self.R += dI  
        
        z = z - np.dot(self.S, np.dot(self.B,self.Q0))
        self.xpred = self.x
        innovation = z - np.dot(self.H, self.xpred)     
        
        
        # ###
        # S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        # K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  
        # self.x = self.x + np.dot(K, innovation)
        # self.infoPpred = np.linalg.inv(self.P)
        # ###
        # if self.timestep <= 1:
        if self.timestep >= 0:
            self.infoPpred = np.linalg.inv(self.P)
            # try:
            #     print(f"z: {z}")
            #     print(f"R {self.R}")
            #     print(f"shaple1: {self.infoPpred.shape}")
            #     print(f"shaple2: {self.H.T.shape}")
            #     print(f"shaple3: {self.R.shape}")
            #     print(f"shaple4: {np.dot(self.H.T, np.linalg.inv(self.R)).shape}")
            #     # # self.infoP = self.infoPpred+ np.dot(np.dot(self.H.T, np.linalg.inv(self.R)),self.H)
            #     if self.R.size == 1:
            #         # if self.R == 0:
            #             # self.R +=0.0001
            #         self.R = self.R.reshape(-1,1)
            #         self.infoP = self.infoPpred+ np.dot(np.dot(self.H.T, 1/(self.R)),self.H)
            # except np.linalg.LinAlgError as e:
            #     print(f"z: {z.shape}")
            #     print(f"R {self.R.shape}")
            #     print(f"shaple1: {self.infoPpred.shape}")
            #     print(f"shaple2: {self.H.T.shape}")
            #     print(f"shaple3: {self.R.shape}")
            #     print(f"shaple4: {np.dot(self.H.T, np.linalg.inv(self.R)).shape}")
            # # self.P = np.linalg.inv(self.infoP)
            
            if self.R.size == 1:
                self.R.reshape(-1,1)
                self.infoP = self.infoPpred+ np.dot(np.dot(self.H.T, 1/(self.R)),self.H)
                self.W = np.dot(np.dot(np.linalg.inv(self.infoP),self.H.T),1/(self.R))
            else:
                self.infoP = self.infoPpred+ np.dot(np.dot(self.H.T, np.linalg.inv(self.R)),self.H)
                self.W = np.dot(np.dot(np.linalg.inv(self.infoP),self.H.T),np.linalg.inv(self.R))

        if inputType is not None:
            self.u, self.u_var = self.input_estimation(z)
        
        # self.x = self.xpred + np.dot(self.W,innovation)
        x_broadcast_ = self.x
        infoP_broadcast_ = self.infoP

        return self.xpred,self.infoPpred,x_broadcast_,infoP_broadcast_
   
    def updateAfterCommunication(self,infoPlist,infoPpredlist,xlist,xpredlist,inputType=None,upredList=None,uVarpreList=None):
        '''
        infoPlist: inverse P list from all nodes including itself, infoP(k+1|k+1)
        infoPpredlist： info P list prediction from all nodes including itself,infoP(k+1|k)
        xlist: state x estimation list from all nodes including itself, x(k+1|k+1)
        xpredlist: x prediction list from all nodes including itself,x(k+1|k)
        '''
        compensation_x = np.zeros_like(self.x)
        # compensation_infoP = np.zeros_like(self.infoP)

        for i in range(len(xlist)):
            # because P won't evolve,so P is prior kwnoledge and all the same:
            compensation_x += np.dot(infoPlist[i],xlist[i]) - np.dot(infoPpredlist[i],xpredlist[i])
            # compensation_infoP += infoPlist[i] - infoPpredlist[i]
           
        #### add compensation
        # self.infoP = self.infoPpred + compensation_infoP 
        # self.P = np.linalg.inv(self.infoP)

        temp = np.dot(self.infoPpred,self.xpred) + compensation_x  
        self.x = np.dot(self.P,temp)

        return        
            
    # def input_estimation(self,z): 
    #     F = np.dot(self.H, self.B)
    #     S = np.dot(np.dot(self.H,self.P),self.H.T)+self.R
    #     M_1 = np.linalg.inv(np.dot(np.dot(F.T,np.linalg.inv(S)),F))
    #     M_2 = np.dot(F.T,np.linalg.inv(S))
    #     M = np.dot(M_1,M_2)
    #     innovation = z - np.dot(self.H, self.x)
    #     u = np.dot(M,innovation)
    #     u_var = M_1

    #     return u, u_var 
    
    def update_discharge(self):
        self.Q0 = np.dot(self.A,self.x) + np.dot(self.B,self.Q0)
    
    
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
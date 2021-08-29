#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os, sys, time, copy
import yaml, h5py, shutil
import scipy
from os import path
from pyDOE import lhs
import cvxpy as cp
import torch
from gpytorch.distributions import MultivariateNormal

from pyTrajectoryUtils.pyTrajectoryUtils.utils import *

def gaussian_sampler(N_sample=100, N=10, gaussian_mean=0.5, gaussian_var=0.1):
    data_t = np.empty((N_sample,N))
    for i in range(N_sample):
        while True:
            x_t = np.random.normal(loc=gaussian_mean, scale=gaussian_var, size=N)
            if np.all(x_t <= 1.0) and np.all(x_t >= 0.0):
                data_t[i,:] = x_t
                break
    return data_t

class TrajSampler():
    def __init__(self, N=10, N_sample=100, x_bound=np.array([0., 1.]), 
                 sigma=50.0, flag_load=False, cov_mode=0, flag_pytorch=True):
        v = np.array([1, -3, 3 ,-1]) # Minimize jerk
        self.N = N        
        if cov_mode == 0:
            A = np.zeros((N+3,N))
            for i in range(N+3):
                for j in range(min(i,N-1),max(i-4,-1),-1):
                    A[i,j] = v[j-i+3]
            R = A.T.dot(A)
            self.cov = np.linalg.inv(R)
        elif cov_mode == 1:
            A = np.zeros((N+3,N))
            for i in range(N+3):
                for j in range(min(i,N-1),max(i-4,-1),-1):
                    A[i,j] = v[j-i+3]
            R = A.T.dot(A)
            X = cp.Variable((N,N), symmetric=True)
            constraints = [X >> np.eye(N)*1e-4]
            constraints += [X[i,i] == 1 for i in range(N)]
            prob = cp.Problem(cp.Minimize(cp.trace(R@X)), constraints)
            prob.solve(solver=cp.CVXOPT)
            self.cov = np.array(X.value)
        
        self.sigma = sigma
        self.x_bound = x_bound
        self.flag_pytorch = flag_pytorch
        if flag_pytorch:
            self.dist = MultivariateNormal(torch.zeros(N),torch.Tensor(self.cov))
        
        self.rand_seed = np.random.get_state()[1][0]
        
    def rsample(self, N_sample=100):
        if self.flag_pytorch:
            x_ret = torch.empty(0,self.N)
        else:
            x_ret = np.empty((0,self.N))
        while x_ret.shape[0] < N_sample:
            N_sample_tmp = np.int(max(self.sigma,1))*N_sample*10
            if self.flag_pytorch:
                x = self.dist.rsample(torch.Size([N_sample_tmp]))
                x_max = torch.min(torch.max(x)/self.sigma,torch.abs(torch.min(x))/self.sigma)
                x_min = -x_max
                accepted = x[(torch.min(x-x_min, axis=1).values>=0.0) & (torch.max(x-x_max, axis=1).values<=0.0)]
                accepted = (accepted-x_min)/(x_max-x_min)*(self.x_bound[1]-self.x_bound[0])+self.x_bound[0]
                x_ret = torch.cat([x_ret,accepted], dim=0)
            else:
                x = np.random.multivariate_normal(np.zeros(self.N), self.cov*self.sigma, size=(N_sample_tmp,))
                x += (self.x_bound[0]+self.x_bound[1])/2
                accepted = x[(np.min(x,axis=1)>=self.x_bound[0]) & (np.max(x,axis=1)<=self.x_bound[1])]
                x_ret = np.concatenate((x_ret, accepted), axis=0)
        x_ret = x_ret[:N_sample, :]
        if self.flag_pytorch:
            x_ret = x_ret.numpy()
        return x_ret

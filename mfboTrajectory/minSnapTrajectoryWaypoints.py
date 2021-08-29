#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os, sys, time, copy
import yaml, h5py, shutil
import scipy
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from os import path
from pyDOE import lhs

from pyTrajectoryUtils.pyTrajectoryUtils.utils import *
from pyTrajectoryUtils.pyTrajectoryUtils.minSnapTrajectory import MinSnapTrajectory
from pyTrajectoryUtils.pyTrajectoryUtils.trajectorySimulation import TrajectorySimulation

class MinSnapTrajectoryWaypoints(MinSnapTrajectory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.yaw_mode = kwargs.get('yaw_mode', 0)
        
        self.snap_acc_obj_ = lambda points, t_set: self.snap_acc_obj(points, t_set, 
                                           yaw_mode=self.yaw_mode,
                                           deg_init_min=0, deg_init_max=4, 
                                           deg_init_yaw_min=0, deg_init_yaw_max=2)
        self.update_traj_ = lambda points, t_set, alpha_set, flag_return_snap=False: \
            self.update_traj(points, t_set, alpha_set, yaw_mode=self.yaw_mode, 
                flag_run_sim=False, flag_return_snap=flag_return_snap)
        return
    
    def wrapper_sanity_check(self, args):
        points = args[0]
        t_set = args[1]
        alpha_set = args[2]
        
        t_set_new = np.multiply(t_set, alpha_set)
        res, d_ordered, d_ordered_yaw = self.snap_acc_obj_(points, t_set_new)
        
        return self.sanity_check(t_set_new, d_ordered, d_ordered_yaw)
    
    def plot_ws(self, t_set, d_ordered, d_ordered_yaw=None):
        flag_loop = self.check_flag_loop(t_set,d_ordered)
        N_POLY = t_set.shape[0]
        
        status = np.zeros((self.N_POINTS*N_POLY,18))

        for der in range(5):
            if flag_loop:
                V_t = self.generate_sampling_matrix_loop(t_set, N=self.N_POINTS, der=der)
            else:
                V_t = self.generate_sampling_matrix(t_set, N=self.N_POINTS, der=der, endpoint=True)
            status[:,3*der:3*(der+1)] = V_t.dot(d_ordered)
        
        if np.all(d_ordered_yaw != None):
            status_yaw_xy = np.zeros((self.N_POINTS*N_POLY,3,2))
            for der in range(3):
                if flag_loop:
                    V_t = self.generate_sampling_matrix_loop_yaw(t_set, N=self.N_POINTS, der=der)
                else:
                    V_t = self.generate_sampling_matrix_yaw(t_set, N=self.N_POINTS, der=der, endpoint=True)
                status_yaw_xy[:,der,:] = V_t.dot(d_ordered_yaw)
            status[:,15:] = self.get_yaw_der(status_yaw_xy)
        
        t_array = np.zeros(self.N_POINTS*N_POLY)
        for i in range(N_POLY):
            for j in range(self.N_POINTS):
                t_array[i*self.N_POINTS+j] = t_array[i*self.N_POINTS+j-1] + t_set[i]/self.N_POINTS
        
        ws, ctrl = self._quadModel.getWs_vector(status)
        for i in range(4):
            fig = plt.figure(figsize=(7,7))
            ax = fig.add_subplot(111)
            ax.plot(t_array, ws[:,i], '-', label='ws {}'.format(i))
            ax.legend()
            ax.grid()
        return
    
    # run simulation with multiple loops & rampin
    def run_sim_loop(self, 
            t_set, d_ordered, d_ordered_yaw,
            flag_debug=False, flag_add_rampin=False, N_loop=1,
            max_pos_err=2.0, max_yaw_err=60, freq_ctrl=200):
        t_set_new = np.tile(t_set,N_loop)
        d_ordered = np.tile(d_ordered,(N_loop,1))
        d_ordered_yaw = np.tile(d_ordered_yaw,(N_loop,1))
        if flag_add_rampin:
            t_set_new, d_ordered, d_ordered_yaw = self.append_rampin(t_set_new, d_ordered, d_ordered_yaw)
        debug_array = self.sim.run_simulation_from_der(t_set_new, d_ordered, d_ordered_yaw, N_trial=1, 
                                                       max_pos_err=max_pos_err, min_pos_err=0.1, 
                                                       max_yaw_err=max_yaw_err, min_yaw_err=15., 
                                                       freq_ctrl=freq_ctrl)
        if flag_debug:
            self.sim.plot_result(debug_array[0], flag_save=False, t_set=t_set_new, d_ordered=d_ordered)
        failure_idx = debug_array[0]["failure_idx"]
        if failure_idx == -1:
            return True
        else:
            return False
#!/usr/bin/env python
# coding: utf-8

import os, sys
import copy
import numpy as np
import pandas as pd
import scipy
from scipy import optimize
from pyDOE import lhs
import h5py
import yaml
import matplotlib.pyplot as plt
from itertools import cycle
from mpl_toolkits.mplot3d import Axes3D

from .quadModel import QuadModel
from .utils import *
from .trajectorySimulation import TrajectorySimulation
import cvxpy as cp

class MinSnapTrajectory(BaseTrajFunc):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if 'cfg_path_sim' in kwargs:
            cfg_path_sim = kwargs['cfg_path_sim']
        else:
            cfg_path_sim = None
        
        if 'drone_model' in kwargs:
            drone_model = kwargs['drone_model']
        else:
            drone_model = None
        
        self._quadModel = QuadModel(cfg_path=cfg_path_sim, drone_model=drone_model)
        self.sim = TrajectorySimulation(*args, **kwargs)
    
    ###############################################################################
    def snap_acc_obj(self, points, t_set, yaw_mode=0, \
                          deg_init_min=0, deg_init_max=4, \
                          deg_end_min=0, deg_end_max=0, \
                          deg_init_yaw_min=0, deg_init_yaw_max=2, \
                          deg_end_yaw_min=0, deg_end_yaw_max=0, \
                          mu=1.0, kt=0):
        points_mean = np.mean(np.abs(points[:,:3]))
        points_mean_global = np.mean(points[:,:3],axis=0)
        flag_loop = self.check_flag_loop(t_set,points[:,:3])
        N_POLY = t_set.shape[0]
        
        pos_obj = lambda x, b: self.snap_obj(x,b,
                                     deg_init_min=deg_init_min,deg_init_max=deg_init_max,
                                     deg_end_min=deg_end_min,deg_end_max=deg_end_max)
        yaw_obj = lambda x, b: self.acc_obj(x,b,
                                     deg_init_min=deg_init_yaw_min,deg_init_max=deg_init_yaw_max,
                                     deg_end_min=deg_end_yaw_min,deg_end_max=deg_end_yaw_max)
        
        b = copy.deepcopy(points[:,:3]-points_mean_global)/points_mean
        res, d_ordered = pos_obj(x=t_set,b=b)
        d_ordered *= points_mean
        if flag_loop:
            for i in range(N_POLY):
                d_ordered[i*self.N_DER,:] += points_mean_global
        else:
            for i in range(N_POLY+1):
                d_ordered[i*self.N_DER,:] += points_mean_global
        
        if yaw_mode == 0:
            b_yaw = np.zeros((points.shape[0],2))
            b_yaw[:,0] = 1
        elif yaw_mode == 1:
            b_yaw = self.get_yaw_forward(t_set, d_ordered)
        elif yaw_mode == 2:
            if points.shape[1] != 4:
                raise("Wrong points format. Append yaw column")
            b_yaw = np.zeros((points.shape[0],2))
            b_yaw[:,0] = np.cos(points[:,-1])
            b_yaw[:,1] = np.sin(points[:,-1])
        else:
            raise("Wrong yaw_mode")
        res_yaw, d_ordered_yaw = yaw_obj(x=t_set, b=b_yaw)
        res += mu*res_yaw
        
        return res, d_ordered, d_ordered_yaw
    
    def update_traj(self, points, t_set, alpha_set, yaw_mode=0, \
                    flag_run_sim=True, flag_return_snap=False):
        t_set_new = np.multiply(t_set, alpha_set)
        
        f_obj = lambda points, t_set: self.snap_acc_obj(points, t_set, yaw_mode=yaw_mode,
                                           deg_init_min=0, deg_init_max=4, 
                                           deg_init_yaw_min=0, deg_init_yaw_max=2)
        _, d_ordered, d_ordered_yaw = f_obj(points, t_set_new)
        
        if flag_run_sim:
            debug_array = self.sim.run_simulation_from_der( \
                t_set=t_set_new, d_ordered=d_ordered, d_ordered_yaw=d_ordered_yaw, \
                max_pos_err=0.2, min_pos_err=0.1, freq_ctrl=200, freq_sim=400)
            self.sim.plot_result(debug_array[0], save_dir="trajectory/result", save_idx="0", t_set=t_set_new, d_ordered=d_ordered)
        
        if flag_return_snap:
            res_i, _, _ = f_obj(points, t_set)
            t_set_scaled = np.multiply(t_set, alpha_set)
            t_set_scaled *= np.sum(t_set)/np.sum(t_set_scaled)
            res_f, _, _ = f_obj(points, t_set_scaled)
            return t_set_new, d_ordered, d_ordered_yaw, res_f/res_i
        else:
            return t_set_new, d_ordered, d_ordered_yaw
    
    def append_rampin(self, t_set, d_ordered, d_ordered_yaw=None, init_points=None, alpha_scale=3.0):
        if np.any(d_ordered_yaw == None):
            yaw_mode = 0
        else:
            yaw_mode = 2
        
        if np.any(init_points == None):
            init_points = np.zeros(4)
            init_points[0] = -2.0
            init_points[1] = 2.0
            init_points[2] = d_ordered[0,2]
        
        points = np.zeros((2,4))
        points[0,:] = init_points
        points[1,:3] = d_ordered[0,:]
        if np.any(init_points == None):
            points[1,3] = np.arctan2(d_ordered_yaw[0,0], d_ordered_yaw[0,1])
        
        t_set_rampin = np.linalg.norm(np.diff(points[:,:3], axis=0),axis=1)*2
        d_ordered_rampin = np.concatenate((
            points[0:1,:3],
            np.zeros((self.N_DER-1,3)),
            d_ordered[:self.N_DER,:]), axis=0)
        d_ordered_yaw_rampin = np.concatenate((
            np.array([[np.cos(points[0,3]),np.sin(points[0,3])]]),
            np.zeros((self.N_DER_YAW-1,2)),
            d_ordered_yaw[:self.N_DER_YAW,:]), axis=0)
        
        t_set_rampin, d_ordered_rampin, d_ordered_yaw_rampin \
            = self.optimize_alpha(points, t_set_rampin, d_ordered_rampin, d_ordered_yaw_rampin, alpha_scale)
        
        flag_loop = self.check_flag_loop(t_set,d_ordered)
        
        t_set = np.append(t_set_rampin, t_set, axis=0)
        
        if flag_loop:
            d_ordered = np.concatenate((
                d_ordered_rampin[:self.N_DER,:], 
                d_ordered,
                d_ordered[:self.N_DER,:]), axis=0)
            d_ordered_yaw = np.concatenate((
                d_ordered_yaw_rampin[:self.N_DER_YAW,:], 
                d_ordered_yaw,
                d_ordered_yaw[:self.N_DER_YAW,:]), axis=0)
        else:
            d_ordered = np.concatenate((
                d_ordered_rampin[:self.N_DER,:], 
                d_ordered), axis=0)
            d_ordered_yaw = np.concatenate((
                d_ordered_yaw_rampin[:self.N_DER_YAW,:], 
                d_ordered_yaw), axis=0)
        
        return t_set, d_ordered, d_ordered_yaw
    
    ###############################################################################
    def sanity_check(self, t_set, d_ordered, d_ordered_yaw=None, flag_parallel=False):
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

        if flag_parallel:
            ws, ctrl = self._quadModel.getWs_vector(status)
            if np.any(ws < self._quadModel.w_min) or np.any(ws > self._quadModel.w_max):
                return False
        else:
            for j in range(self.N_POINTS*N_POLY):
                ws, ctrl = self._quadModel.getWs(status[j,:])
                if np.any(ws < self._quadModel.w_min) or np.any(ws > self._quadModel.w_max):
                    return False

        return True
    
    ###############################################################################
    def snap_obj(self, x, b, b_ext_init=None, b_ext_end=None, 
                 deg_init_min=0, deg_init_max=4, deg_end_min=0, deg_end_max=0, kt=0):
        flag_loop = self.check_flag_loop(x,b)
        N_POLY = x.shape[0]
        if np.any(b_ext_init == None):
            b_ext_init = np.zeros((deg_init_max-deg_init_min,b.shape[1]))
        else:
            assert b_ext_init.shape[0] == (deg_init_max-deg_init_min)
            assert b_ext_init.shape[1] == b.shape[1]
        if np.any(b_ext_end == None):
            b_ext_end = np.zeros((deg_end_max-deg_end_min,b.shape[1]))
        else:
            assert b_ext_end.shape[0] == (deg_end_max-deg_end_min)
            assert b_ext_end.shape[1] == b.shape[1]
        
        if flag_loop:
            P = self.generate_perm_matrix(x.shape[0]-1, self.N_DER)
            A_sys = self.generate_sampling_matrix_loop(x, N=self.N_POINTS, der=self.MAX_SYS_DEG)
        else:
            P = self.generate_perm_matrix(x.shape[0], self.N_DER)
            A_sys = self.generate_sampling_matrix(x, N=self.N_POINTS, der=self.MAX_SYS_DEG, endpoint=True)
        D_tw = self.generate_weight_matrix(x, self.N_POINTS)
        
        A_sys_t = A_sys.dot(P.T)[:,:]
        R = A_sys_t.T.dot(D_tw).dot(A_sys_t)
        if flag_loop:
            R_idx = np.concatenate((np.arange(b.shape[0],b.shape[0]+deg_init_min),
                                    np.arange(b.shape[0]+deg_init_max,R.shape[0])), axis=0)
        else:
            R_idx = np.concatenate((np.arange(b.shape[0],b.shape[0]+deg_init_min),
                                    np.arange(b.shape[0]+deg_init_max,R.shape[0]-self.N_DER+1+deg_end_min),
                                    np.arange(R.shape[0]-self.N_DER+1+deg_end_max,R.shape[0])), axis=0)
        Rpp = R[np.ix_(R_idx,R_idx)]
        Rpp_inv = self.get_matrix_inv(Rpp)

        d_p = -Rpp_inv.dot(R[np.ix_(np.arange(b.shape[0]),R_idx)].T).dot(b)
        if flag_loop:
            d_tmp = np.concatenate((b,
                                  d_p[:deg_init_min,:],
                                  b_ext_init,
                                  d_p[deg_init_min:,:]),axis=0)
        else:
            d_tmp = np.concatenate((b,
                                  d_p[:deg_init_min,:],
                                  b_ext_init,
                                  d_p[deg_init_min:d_p.shape[0]-self.N_DER+1+deg_end_max,:],
                                  b_ext_end,
                                  d_p[d_p.shape[0]-self.N_DER+1+deg_end_max:,:]),axis=0)

        res = np.trace(d_tmp.T.dot(R).dot(d_tmp)) + kt*np.sum(x)
        d_ordered = P.T.dot(d_tmp)
        
        if res < 0:
            res = 1e10
        return res, d_ordered
    
    def acc_obj(self, x, b, b_ext_init=None, b_ext_end=None, 
                 deg_init_min=0, deg_init_max=2, deg_end_min=0, deg_end_max=0):
        flag_loop = self.check_flag_loop(x,b)
        N_POLY = x.shape[0]
        if np.any(b_ext_init == None):
            b_ext_init = np.zeros((deg_init_max-deg_init_min,b.shape[1]))
        else:
            assert b_ext_init.shape[0] == (deg_init_max-deg_init_min)
            assert b_ext_init.shape[1] == b.shape[1]
        if np.any(b_ext_end == None):
            b_ext_end = np.zeros((deg_end_max-deg_end_min,b.shape[1]))
        else:
            assert b_ext_end.shape[0] == (deg_end_max-deg_end_min)
            assert b_ext_end.shape[1] == b.shape[1]
        
        if flag_loop:
            P = self.generate_perm_matrix(x.shape[0]-1, self.N_DER_YAW)
            A_sys = self.generate_sampling_matrix_loop_yaw(x, N=self.N_POINTS, der=self.MAX_SYS_DEG_YAW)
        else:
            P = self.generate_perm_matrix(x.shape[0], self.N_DER_YAW)
            A_sys = self.generate_sampling_matrix_yaw(x, N=self.N_POINTS, der=self.MAX_SYS_DEG_YAW, endpoint=True)
        D_tw = self.generate_weight_matrix(x, self.N_POINTS)

        R = P.dot(A_sys.T).dot(D_tw).dot(A_sys).dot(P.T)
        if flag_loop:
            R_idx = np.concatenate((np.arange(b.shape[0],b.shape[0]+deg_init_min),
                                    np.arange(b.shape[0]+deg_init_max,R.shape[0])), axis=0)
        else:
            R_idx = np.concatenate((np.arange(b.shape[0],b.shape[0]+deg_init_min),
                                    np.arange(b.shape[0]+deg_init_max,R.shape[0]-self.N_DER+1+deg_end_min),
                                    np.arange(R.shape[0]-self.N_DER+1+deg_end_max,R.shape[0])), axis=0)
        Rpp = R[np.ix_(R_idx,R_idx)]
        Rpp_inv = self.get_matrix_inv(Rpp)
        d_p = -Rpp_inv.dot(R[np.ix_(np.arange(b.shape[0]),R_idx)].T).dot(b)
        if flag_loop:
            d_tmp = np.concatenate((b,
                                  d_p[:deg_init_min,:],
                                  b_ext_init,
                                  d_p[deg_init_min:,:]),axis=0)
        else:
            d_tmp = np.concatenate((b,
                                  d_p[:deg_init_min,:],
                                  b_ext_init,
                                  d_p[deg_init_min:d_p.shape[0]-self.N_DER_YAW+1+deg_end_max,:],
                                  b_ext_end,
                                  d_p[d_p.shape[0]-self.N_DER_YAW+1+deg_end_max:,:]),axis=0)
        res = np.trace(d_tmp.T.dot(R).dot(d_tmp))
        d_ordered = P.T.dot(d_tmp)
        
        if res < 0:
            res = 1e10
        return res, d_ordered
    
    def get_yaw_forward(self, t_set, d_ordered):
        flag_loop = self.check_flag_loop(t_set,d_ordered)
        N_POLY = t_set.shape[0]
        if not flag_loop:
            N_POLY += 1
        
        yaw_ref = np.zeros((N_POLY,2))
        vel = np.zeros((N_POLY,2))
        if flag_loop:
            V_t = self.generate_sampling_matrix_loop(t_set, N=self.N_POINTS, der=1)
        else:
            V_t = self.generate_sampling_matrix(t_set, N=self.N_POINTS, der=1, endpoint=True)
        vel[:np.int(V_t.shape[0]/self.N_POINTS),:] = (V_t.dot(d_ordered[:,:2]))[::self.N_POINTS,:]
        
        if not flag_loop:
            vel[-1,:] = (V_t.dot(d_ordered[:,:2]))[-1,:]
        
        for i in range(N_POLY):
            if np.linalg.norm(vel[i,:2]) < 1e-6:
                if i < N_POLY:
                    for j in range(1,self.N_POINTS):
                        vel_tmp = V_t.dot(d_ordered[:,:2])[i*self.N_POINTS+j,:]
                        if np.linalg.norm(vel_tmp[:2]) > 1e-6:
                            vel_tmp2 = vel_tmp
                            break
                        vel_tmp2 = np.array([1,0])
                else:
                    for j in range(1,self.N_POINTS):
                        vel_tmp = V_t.dot(d_ordered[:,:2])[i*self.N_POINTS-j,:]
                        if np.linalg.norm(vel_tmp[:2]) > 1e-6:
                            vel_tmp2 = vel_tmp
                            break
                        vel_tmp2 = np.array([1,0])
                yaw_ref[i,0] = vel_tmp2[0]/np.linalg.norm(vel_tmp2[:2])
                yaw_ref[i,1] = vel_tmp2[1]/np.linalg.norm(vel_tmp2[:2])
            else:
                yaw_ref[i,0] = vel[i,0]/np.linalg.norm(vel[i,:2])
                yaw_ref[i,1] = vel[i,1]/np.linalg.norm(vel[i,:2])

        if np.any(np.abs(np.linalg.norm(yaw_ref, axis=1)-1)>1e-3):
            print("Wrong yaw forward")
            sys.exit(0)
            
        return yaw_ref
    
    def optimize_alpha(self, points, t_set, d_ordered, d_ordered_yaw, alpha_scale=1.0, sanity_check_t=None, flag_return_alpha=False):
        if sanity_check_t == None:
            sanity_check_t = self.sanity_check
        
        # Optimizae alpha
        alpha = 2.0
        dalpha = 0.1
        alpha_tmp = alpha
        t_set_ret = copy.deepcopy(t_set)
        d_ordered_ret = copy.deepcopy(d_ordered)
        d_ordered_yaw_ret = copy.deepcopy(d_ordered_yaw)
        
        def get_alpha_matrix(alpha):
            T_alpha = np.diag(self.generate_basis(1./alpha,self.N_DER-1,0))
            T_alpha_all = np.zeros((self.N_DER*points.shape[0],self.N_DER*points.shape[0]))
            for i in range(points.shape[0]):
                T_alpha_all[i*self.N_DER:(i+1)*self.N_DER,i*self.N_DER:(i+1)*self.N_DER] = T_alpha
            return T_alpha_all
        
        def get_alpha_matrix_yaw(alpha):
            T_alpha = np.diag(self.generate_basis(1./alpha,self.N_DER_YAW-1,0))
            T_alpha_all = np.zeros((self.N_DER_YAW*points.shape[0],self.N_DER_YAW*points.shape[0]))
            for i in range(points.shape[0]):
                T_alpha_all[i*self.N_DER_YAW:(i+1)*self.N_DER_YAW,i*self.N_DER_YAW:(i+1)*self.N_DER_YAW] = T_alpha
            return T_alpha_all
        
        while True:
            t_set_opt = t_set * alpha
            d_ordered_opt = get_alpha_matrix(alpha).dot(d_ordered)
            d_ordered_yaw_opt = get_alpha_matrix_yaw(alpha).dot(d_ordered_yaw)
            
            if not sanity_check_t(t_set_opt, d_ordered_opt, d_ordered_yaw_opt):
                alpha += 1.0
            else:
                break
            
        while True:
            alpha_tmp = alpha - dalpha
            t_set_opt = t_set * alpha_tmp
            d_ordered_opt = get_alpha_matrix(alpha_tmp).dot(d_ordered)
            d_ordered_yaw_opt = get_alpha_matrix_yaw(alpha_tmp).dot(d_ordered_yaw)
            
            if not sanity_check_t(t_set_opt, d_ordered_opt, d_ordered_yaw_opt):
                dalpha *= 0.1
            else:
                alpha = alpha_tmp
                t_set_ret = t_set_opt
                d_ordered_ret = d_ordered_opt
                d_ordered_yaw_ret = d_ordered_yaw_opt
            
            if dalpha < 1e-5 or alpha < 1e-5:
#                 print("Optimize alpha: {}".format(alpha))
                break
        
        t_set = t_set_ret * alpha_scale
        d_ordered = get_alpha_matrix(alpha_scale).dot(d_ordered_ret)
        d_ordered_yaw = get_alpha_matrix_yaw(alpha_scale).dot(d_ordered_yaw_ret)
        
        if flag_return_alpha:
            return t_set, d_ordered, d_ordered_yaw, alpha
        else:
            return t_set, d_ordered, d_ordered_yaw
    
    ###############################################################################
    def get_min_snap_traj(self, points, alpha_scale=1.0, flag_loop=False, yaw_mode=0, \
                          deg_init_min=0, deg_init_max=4, \
                          deg_end_min=0, deg_end_max=0, \
                          deg_init_yaw_min=0, deg_init_yaw_max=2, \
                          deg_end_yaw_min=0, deg_end_yaw_max=0, \
                          mu=1.0, kt=0, \
                          flag_rand_init=False, flag_numpy_opt=False):
        points_mean = np.mean(np.abs(points[:,:3]))
        points_mean_global = np.mean(points[:,:3],axis=0)
        
        pos_obj = lambda x, b: self.snap_obj(x,b,
                                     deg_init_min=deg_init_min,deg_init_max=deg_init_max,
                                     deg_end_min=deg_end_min,deg_end_max=deg_end_max)
        yaw_obj = lambda x, b: self.acc_obj(x,b,
                                     deg_init_min=deg_init_yaw_min,deg_init_max=deg_init_yaw_max,
                                     deg_end_min=deg_end_yaw_min,deg_end_max=deg_end_yaw_max)
        
        if flag_loop:
            N_POLY = points.shape[0]
        else:
            N_POLY = points.shape[0]-1
        t_set = np.linalg.norm(np.diff(points[:,:3], axis=0),axis=1)*2/points_mean
        if flag_loop:
            t_set = np.append(t_set,np.linalg.norm(points[-1,:3]-points[0,:3])*2/points_mean)
        b = (copy.deepcopy(points[:,:3])-points_mean_global)/points_mean
        
        MAX_ITER = 500
        lr = 0.5*N_POLY
        dt = 1e-3
        
        if yaw_mode == 0 or yaw_mode == 1:
            b_yaw = np.zeros((points.shape[0],2))
            b_yaw[:,0] = 1
        elif yaw_mode == 2:
            if points.shape[1] != 4:
                print("Wrong points format. Append yaw column")
            b_yaw = np.zeros((points.shape[0],2))
            b_yaw[:,0] = np.cos(points[:,-1])
            b_yaw[:,1] = np.sin(points[:,-1])
        else:
            raise("Wrong yaw_mode")
        
        def f_obj(x):
            if kt == 0:
                x_t = x*1.0*np.shape(x)[0]/np.sum(x)
            else:
                x_t = x*1.0
            f0, d_ordered = pos_obj(x=x_t, b=b)
            if yaw_mode == 0:
                b_yaw = np.zeros((points.shape[0],2))
                b_yaw[:,0] = 1
            elif yaw_mode == 1:
                b_yaw = self.get_yaw_forward(x_t, d_ordered)
            elif yaw_mode == 2:
                if points.shape[1] != 4:
                    print("Wrong points format. Append yaw column")
                b_yaw = np.zeros((points.shape[0],2))
                b_yaw[:,0] = np.cos(points[:,-1])
                b_yaw[:,1] = np.sin(points[:,-1])
            else:
                raise("Wrong yaw_mode")
            f0_yaw, _ = yaw_obj(x=x_t, b=b_yaw)
            f0 += mu*f0_yaw + kt*np.sum(x_t)
            return f0
        
        print("t_set (initial) : {}".format(t_set))
        
        if flag_rand_init:
            # Random init
            N_rand = 1000
            alpha_set_tmp = lhs(t_set.shape[0], N_rand)*1.8+0.1
            min_t_set = None
            min_f_obj = -1
            for i in range(N_rand):
                t_set_tmp = np.multiply(t_set, alpha_set_tmp[i,:])
                t_set_tmp *= np.sum(t_set)/np.sum(t_set_tmp)
                f_obj_tmp = f_obj(t_set_tmp)
                if (min_f_obj == -1 or min_f_obj > f_obj_tmp) and f_obj_tmp > 0:
                    min_f_obj = f_obj_tmp
                    min_t_set = t_set_tmp
            t_set = min_t_set
            print("t_set (rand_init): {}".format(t_set))
        
        if flag_numpy_opt:
            # Optimizae time ratio
            for t in range(MAX_ITER):
                grad = np.zeros(N_POLY)

                f0, d_ordered = pos_obj(x=t_set, b=b)
                if yaw_mode == 1:
                    b_yaw = self.get_yaw_forward(t_set, d_ordered)
                elif yaw_mode == 2:
                    if points.shape[1] != 4:
                        print("Wrong points format. Append yaw column")
                    b_yaw[:,0] = np.cos(points[:,-1])
                    b_yaw[:,1] = np.sin(points[:,-1])
                f0_yaw, _ = yaw_obj(x=t_set, b=b_yaw)
                f0 += mu*f0_yaw

                for i in range(N_POLY):
                    t_set_tmp = copy.deepcopy(t_set)
                    t_set_tmp[i] += dt

                    f1, _ = pos_obj(x=t_set_tmp, b=b)
                    f1_yaw, _ = yaw_obj(x=t_set, b=b_yaw)
                    f1 += mu*f1_yaw
                    grad[i] = (f1-f0)/dt

                err = np.mean(np.abs(grad))
                grad /= np.linalg.norm(grad)

                t_set_tmp = t_set-lr*grad

                if np.any(t_set_tmp < 0.0):
                    lr *= 0.1
                    continue

                f_tmp, d_ordered = pos_obj(x=t_set_tmp*np.sum(t_set)/np.sum(t_set_tmp), b=b)
                if f0 > 0:
                    f_ratio = f_tmp/f0
                else:
                    raise("Wrong overall snaps")
                    f_ratio = 0
                
                if lr < 1e-20 and f_ratio < 1e-2:
                    break

                t_set -= lr*grad

                if err < 1e-3 and f_ratio < 1e-2:
                    break
            print("t_set (numpy_opt): {}".format(t_set))
        
        bounds = []
        for i in range(t_set.shape[0]):
            bounds.append((0.001, 1000.0))
        
        res_x, res_f, res_d = scipy.optimize.fmin_l_bfgs_b(\
                                    f_obj, x0=t_set, bounds=bounds, \
                                    approx_grad=True, epsilon=1e-4, maxiter=MAX_ITER, \
                                    iprint=1)
        t_set = np.array(res_x)
        
        x_t = t_set*1.0*np.shape(t_set)[0]/np.sum(t_set)
        rel_snap, _ = pos_obj(x=x_t, b=b)
        print("t_set (final): {}".format(t_set))
        print("Relative snap: {}".format(rel_snap))

        _, d_ordered = pos_obj(t_set, b=b)
        d_ordered *= points_mean
        if flag_loop:
            for i in range(N_POLY):
                d_ordered[i*self.N_DER,:] += points_mean_global
        else:
            for i in range(N_POLY+1):
                d_ordered[i*self.N_DER,:] += points_mean_global
        
        if yaw_mode == 1:
            b_yaw = self.get_yaw_forward(t_set, d_ordered)
        _, d_ordered_yaw = yaw_obj(x=t_set, b=b_yaw)
        
        return self.optimize_alpha(points, t_set, d_ordered, d_ordered_yaw, alpha_scale)
    
if __name__ == "__main__":
    poly = MinSnapTraj(MAX_POLY_DEG = 9, MAX_SYS_DEG = 4, N_POINTS = 40)
    
    

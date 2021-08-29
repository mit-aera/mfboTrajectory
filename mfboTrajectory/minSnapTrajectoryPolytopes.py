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
from cvxpy import *
import cvxpy as cp
import plotly
import plotly.graph_objects as go

from pyTrajectoryUtils.pyTrajectoryUtils.quadModel import QuadModel
from pyTrajectoryUtils.pyTrajectoryUtils.utils import *
from pyTrajectoryUtils.pyTrajectoryUtils.minSnapTrajectory import MinSnapTrajectory

class MinSnapTrajectoryPolytopes(MinSnapTrajectory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.yaw_mode = kwargs.get('yaw_mode', 0)
        self.qp_optimizer = kwargs.get('qp_optimizer', 'osqp')
        self.default_t_set_scale = 1.0
        
    ###############################################################################
    def save_trajectory_yaml(self, t_set, d_ordered, d_ordered_yaw=None, \
                         traj_dir='./trajectory', traj_name="test"):
        
        poly_coeff, poly_coeff_yaw = self.der_to_poly(t_set, d_ordered, d_ordered_yaw)
        
#         print("\nContinuity checking")
#         print(poly_coeff.shape)
        for poly_ii in range(poly_coeff.shape[0]-1):
            for der_ii in range(5):
                basis_curr = self.generate_basis(t_set[poly_ii],self.MAX_POLY_DEG, der_ii)
                basis_next = self.generate_basis(0.0,self.MAX_POLY_DEG, der_ii)
                val_curr = basis_curr.dot(poly_coeff[poly_ii, :, :])
                val_next = basis_next.dot(poly_coeff[poly_ii+1, :, :])
                if np.any((np.abs(val_curr - val_next) > 1e-3)):
                    print("[ERROR] poly_ii: {}, der_ii: {}".format(poly_ii, der_ii))
                    print(val_curr)
                    print(val_next)

        dim_prefix = ["x","y","z","yaw_c","yaw_s"]
        yamlFile = os.path.join(traj_dir,'{}.yaml'.format(traj_name))
        yaml_out = open(yamlFile,"w")
        yaml_out.write("coeff:\n")
        for dim_ii in range(poly_coeff.shape[2]):
            yaml_out.write("  {}:\n".format(dim_prefix[dim_ii]))
            for poly_idx in range(poly_coeff.shape[0]):
                yaml_out.write("    - [{}]\n".format(','.join([str(x) for x in poly_coeff[poly_idx,:,dim_ii]])))
            yaml_out.write("\n")
        
        yaw = np.zeros((self.MAX_POLY_DEG_YAW+1,2))
        yaw[0,0] = 1.0
        for dim_ii in range(3,5):
            yaml_out.write("  {}:\n".format(dim_prefix[dim_ii]))
            
            for poly_idx in range(poly_coeff.shape[0]):
                if np.all(d_ordered_yaw != None):
                    yaw = poly_coeff_yaw[poly_idx,:,:]
                yaml_out.write("    - [{}]\n".format(','.join([str(x) for x in yaw[:,dim_ii-3]])))
            yaml_out.write("\n")
        yaml_out.write("dt: [{}]\n".format(','.join([str(x) for x in t_set])))
        yaml_out.close()
        return
    
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
    def generate_sum_matrix(self, x, der=4, flag_loop=False):
        Q = np.zeros((self.MAX_POLY_DEG+1,self.MAX_POLY_DEG+1))
        for i in range(self.MAX_POLY_DEG+1):
            for j in range(self.MAX_POLY_DEG+1):
                if i >= der and j >= der:
                    Q[i,j] = factorial(i)/factorial(i-der)*factorial(j)/factorial(j-der)/(i+j+1-2*der)
        
        V = np.hstack([self.v0_mat[0,:,:].T, self.v1_mat[0,:,:].T])
        M = V.T.dot(Q).dot(V)
        W0 = M[:self.N_DER,:self.N_DER]
        W1 = M[:self.N_DER,self.N_DER:]
        W1T = M[self.N_DER:,:self.N_DER]
        W2 = M[self.N_DER:,self.N_DER:]
        
        N_POLY = x.shape[0]
        if flag_loop:
            Mat_obj = np.zeros((self.N_DER*N_POLY,self.N_DER*N_POLY))
        else:
            Mat_obj = np.zeros((self.N_DER*(N_POLY+1),self.N_DER*(N_POLY+1)))
        for i in range(N_POLY):
            if flag_loop:
                i_n = (i+1)%(N_POLY)
            else:
                i_n = (i+1)
            T_mat = np.diag(self.generate_basis(x[i],self.N_DER-1,0))
            Mat_obj[i*self.N_DER:(i+1)*self.N_DER, \
                    i*self.N_DER:(i+1)*self.N_DER] \
                += T_mat.dot(W0).dot(T_mat)/(x[i]**(2*der-1))
            Mat_obj[i*self.N_DER:(i+1)*self.N_DER, \
                    i_n*self.N_DER:(i_n+1)*self.N_DER] \
                += T_mat.dot(W1).dot(T_mat)/(x[i]**(2*der-1))
            Mat_obj[i_n*self.N_DER:(i_n+1)*self.N_DER, \
                    i*self.N_DER:(i+1)*self.N_DER] \
                += T_mat.dot(W1.T).dot(T_mat)/(x[i]**(2*der-1))
            Mat_obj[i_n*self.N_DER:(i_n+1)*self.N_DER, \
                    i_n*self.N_DER:(i_n+1)*self.N_DER] \
                += T_mat.dot(W2).dot(T_mat)/(x[i]**(2*der-1))
            
        return Mat_obj
    
    def generate_sum_matrix_yaw(self, x, der=2, flag_loop=False):
        Q = np.zeros((self.MAX_POLY_DEG_YAW+1,self.MAX_POLY_DEG_YAW+1))
        for i in range(self.MAX_POLY_DEG_YAW+1):
            for j in range(self.MAX_POLY_DEG_YAW+1):
                if i >= der and j >= der:
                    Q[i,j] = 1/(i+j+1-2*der)*factorial(i)/factorial(i-der)*factorial(j)/factorial(j-der)
        
        V = np.hstack([self.v0_mat_yaw[0,:,:].T, self.v1_mat_yaw[0,:,:].T])
        M = V.T.dot(Q).dot(V)
        W0 = M[:self.N_DER_YAW,:self.N_DER_YAW]
        W1 = M[:self.N_DER_YAW,self.N_DER_YAW:]
        W2 = M[self.N_DER_YAW:,self.N_DER_YAW:]
        
        N_POLY = x.shape[0]
        if flag_loop:
            Mat_obj = np.zeros((self.N_DER_YAW*N_POLY,self.N_DER_YAW*N_POLY))
        else:
            Mat_obj = np.zeros((self.N_DER_YAW*(N_POLY+1),self.N_DER_YAW*(N_POLY+1)))
        for i in range(N_POLY):
            if flag_loop:
                i_n = (i+1)%(N_POLY)
            else:
                i_n = (i+1)
            T_mat = np.diag(self.generate_basis(x[i],self.N_DER_YAW-1,0))
            Mat_obj[i*self.N_DER_YAW:(i+1)*self.N_DER_YAW, \
                    i*self.N_DER_YAW:(i+1)*self.N_DER_YAW] \
                += T_mat.dot(W0).dot(T_mat)/(x[i]**(2*der-1))
            Mat_obj[i*self.N_DER_YAW:(i+1)*self.N_DER_YAW, \
                    i_n*self.N_DER_YAW:(i_n+1)*self.N_DER_YAW] \
                += T_mat.dot(W1).dot(T_mat)/(x[i]**(2*der-1))
            Mat_obj[i_n*self.N_DER_YAW:(i_n+1)*self.N_DER_YAW, \
                    i*self.N_DER_YAW:(i+1)*self.N_DER_YAW] \
                += T_mat.dot(W1.T).dot(T_mat)/(x[i]**(2*der-1))
            Mat_obj[i_n*self.N_DER_YAW:(i_n+1)*self.N_DER_YAW, \
                    i_n*self.N_DER_YAW:(i_n+1)*self.N_DER_YAW] \
                += T_mat.dot(W2).dot(T_mat)/(x[i]**(2*der-1))

        return Mat_obj
    
    def snap_obj(self, t_set, points, plane_pos_set, \
                 deg_init_min=0, deg_init_max=4, deg_end_min=0, deg_end_max=0, \
                 flag_init_point=True, flag_fixed_point=False, flag_fixed_end_point=False):
        
        N_wp = points.shape[0]
        flag_loop = self.check_flag_loop(t_set,points)
        N_POLY = t_set.shape[0]
        
        t_scale_list = [self.default_t_set_scale]
        for i in range(1,3):
            t_scale_list.append(self.default_t_set_scale*np.power(2,i))
            t_scale_list.append(self.default_t_set_scale/np.power(2,i))
        
        t_set_scale = self.default_t_set_scale
        for t_scale in t_scale_list:
            t_set_scale = t_scale
            x_t = t_set*t_set_scale

            Q_pos = self.generate_sum_matrix(x_t, der=4, flag_loop=flag_loop)    
            Q_obj = scipy.linalg.block_diag(Q_pos,Q_pos,Q_pos)

            if flag_loop:
                V_t = self.generate_sampling_matrix(x_t, N=self.N_POINTS, der=0, endpoint=False)
            else:
                V_t = self.generate_sampling_matrix(x_t, N=self.N_POINTS, der=0, endpoint=True)

            # Create two scalar optimization variables.
            if flag_loop:
                n = N_POLY*self.N_DER
            else:
                n = (N_POLY+1)*self.N_DER
            x = Variable(n*3)
            constraints = []

            # Corridor constraints
            for p_ii in range(len(plane_pos_set)):
                if flag_loop:
                    p_ii_n = (p_ii+1)%(len(plane_pos_set))
                else:
                    p_ii_n = (p_ii+1)

                # Add plane contraints
                V_t_tmp = V_t[p_ii*self.N_POINTS:(p_ii+1)*self.N_POINTS,:]
                for i in range(len(plane_pos_set[p_ii]["constraints_plane"])):
                    v0 = np.array(plane_pos_set[p_ii]["constraints_plane"][i][0])
                    v1 = np.array(plane_pos_set[p_ii]["constraints_plane"][i][1])
                    for k in range(2,len(plane_pos_set[p_ii]["constraints_plane"][i])):
                        v2 = np.array(plane_pos_set[p_ii]["constraints_plane"][i][k])
                        res_t = np.fabs((v1-v0).dot(v2-v0)/np.linalg.norm(v1-v0)/np.linalg.norm(v2-v0))
                        if res_t < 1-1e-4:
                            break
                    V_norm = np.cross(v1-v0, v2-v0)*1.0
                    V_norm /= np.linalg.norm(V_norm)
                    V_bias = V_norm.dot(v0)
                    A = np.hstack([V_norm[0]*V_t_tmp,V_norm[1]*V_t_tmp,V_norm[2]*V_t_tmp])
                    constraints.append(A*x >= V_bias)
                
                # Corner boundary
                for i in range(len(plane_pos_set[p_ii]["corner_plane"])):
                    v0 = np.array(plane_pos_set[p_ii]["corner_plane"][i][0])
                    v1 = np.array(plane_pos_set[p_ii]["corner_plane"][i][1])
                    v2 = np.array(plane_pos_set[p_ii]["corner_plane"][i][2])
                    V_norm = np.cross(v1-v0, v2-v0)*1.0
                    V_norm /= np.linalg.norm(V_norm)
                    V_bias = V_norm.dot(v0)
                    A = np.hstack([V_norm[0]*V_t_tmp,V_norm[1]*V_t_tmp,V_norm[2]*V_t_tmp])
                    constraints.append(A*x >= V_bias)

                # Init end point boundary
                if len(plane_pos_set[p_ii]["input_plane"]) > 0:
                    v0 = np.array(plane_pos_set[p_ii]["input_plane"][0])
                    v1 = np.array(plane_pos_set[p_ii]["input_plane"][1])
                    v2 = np.array(plane_pos_set[p_ii]["input_plane"][2])
                    V_norm = np.cross(v1-v0, v2-v0)*1.0
                    V_norm /= np.linalg.norm(V_norm)
                    V_bias = V_norm.dot(v0)
                    A = np.hstack([V_norm[0]*V_t_tmp,V_norm[1]*V_t_tmp,V_norm[2]*V_t_tmp])
                    constraints.append(A*x >= V_bias)

                # Init end point boundary
                if len(plane_pos_set[p_ii]["input_plane"]) > 0:
                    v0 = np.array(plane_pos_set[p_ii]["input_plane"][0])
                    v1 = np.array(plane_pos_set[p_ii]["input_plane"][1])
                    v2 = np.array(plane_pos_set[p_ii]["input_plane"][2])
                    V_norm = np.cross(v1-v0, v2-v0)*1.0
                    V_norm /= np.linalg.norm(V_norm)
                    V_bias = V_norm.dot(v0)
                    A = np.hstack([V_norm[0]*V_t_tmp,V_norm[1]*V_t_tmp,V_norm[2]*V_t_tmp])
                    constraints.append(A*x >= V_bias)

                # Final end point boundary
                if len(plane_pos_set[p_ii]["output_plane"]) > 0:
                    v0 = np.array(plane_pos_set[p_ii]["output_plane"][0])
                    v1 = np.array(plane_pos_set[p_ii]["output_plane"][1])
                    v2 = np.array(plane_pos_set[p_ii]["output_plane"][2])
                    V_norm = np.cross(v1-v0, v2-v0)*1.0
                    V_norm /= np.linalg.norm(V_norm)
                    V_bias = V_norm.dot(v0)
                    A = np.hstack([V_norm[0]*V_t_tmp,V_norm[1]*V_t_tmp,V_norm[2]*V_t_tmp])
                    constraints.append(A*x >= V_bias)

                # Final plane end point
                if not flag_loop and p_ii == N_POLY-1:
                    V_t_fixed = V_t[-1:,:]
                else:
                    V_t_fixed = V_t[(p_ii+1)*self.N_POINTS:(p_ii+1)*self.N_POINTS+1,:]
                if flag_fixed_point:
                    A = scipy.linalg.block_diag(V_t_fixed,V_t_fixed,V_t_fixed)
                    constraints.append(A*x == points[p_ii_n,:3])
                else:
                    if p_ii == len(plane_pos_set)-1:
                        if flag_fixed_end_point:
                            A = scipy.linalg.block_diag(V_t_fixed,V_t_fixed,V_t_fixed)
                            constraints.append(A*x == points[p_ii_n,:3])
                    else:
                        v0 = np.array(plane_pos_set[p_ii]["output_plane"][0])
                        v1 = np.array(plane_pos_set[p_ii]["output_plane"][1])
                        v2 = np.array(plane_pos_set[p_ii]["output_plane"][2])
                        V_norm = np.cross(v1-v0, v2-v0)*1.0
                        V_norm /= np.linalg.norm(V_norm)
                        V_bias = V_norm.dot(v0)                    
                        A = np.hstack([V_norm[0]*V_t_fixed,V_norm[1]*V_t_fixed,V_norm[2]*V_t_fixed])
                        constraints.append(A*x == V_bias)

            # Starting point constraints
            if flag_init_point:
                start_point = points[0,:]
                constraints.append(x[0] == start_point[0])
                constraints.append(x[n] == start_point[1])
                constraints.append(x[2*n] == start_point[2])

#             if (flag_fixed_end_point or flag_fixed_point) and (not flag_loop):
#                 end_point = points[-1,:]
#                 constraints.append(x[N_POLY*self.N_DER] == end_point[0])
#                 constraints.append(x[n+N_POLY*self.N_DER] == end_point[1])
#                 constraints.append(x[2*n+N_POLY*self.N_DER] == end_point[2])
                
            # End point higher derivative constraints
            if deg_init_max > deg_init_min:
                deg_min_t = np.int(np.maximum(1, deg_init_min))
                constraints.append(x[deg_min_t:deg_init_max+1] == 0)
                constraints.append(x[n+deg_min_t:n+deg_init_max+1] == 0)
                constraints.append(x[2*n+deg_min_t:2*n+deg_init_max+1] == 0)

            if not flag_loop and deg_end_max > deg_end_min:
                deg_min_t = np.int(np.maximum(1, deg_end_min))
                n_t = N_POLY*self.N_DER
                constraints.append(x[n_t+deg_min_t:n_t+deg_end_max+1] == 0)
                constraints.append(x[n_t+n+deg_min_t:n_t+n+deg_end_max+1] == 0)
                constraints.append(x[n_t+2*n+deg_min_t:n_t+2*n+deg_end_max+1] == 0)

            # Form objective.
            obj = Minimize(0.5*quad_form(x, Q_obj))

            # Form and solve problem.
            prob = Problem(obj, constraints)
            try:
                if self.qp_optimizer == 'osqp':
                    prob.solve(solver=cp.OSQP, verbose=False)
                elif self.qp_optimizer == 'gurobi':
                    prob.solve(solver=cp.GUROBI, verbose=False)
                elif self.qp_optimizer == 'cvxopt':
                    prob.solve(solver=cp.CVXOPT, verbose=False)
                else:
                    continue
            except cp.error.DCPError:
                continue
            except cp.error.SolverError:
                continue

            if prob.status not in ["infeasible", "unbounded"] and np.all(x.value != None):
                if self.default_t_set_scale != t_scale:
                    prRed("[snap_obj] Update t_scale from {} to {}.".format(self.default_t_set_scale, t_scale))
                    # self.default_t_set_scale = t_scale
                break
            else:
                prRed(prob.status)
        
        if prob.status in ["infeasible", "unbounded"]:
            prRed("[snap_obj] Failed to optimize t_set")
            return
        elif np.any(x.value == None):
            prRed("[snap_obj] x value is None")
            d_ordered = np.zeros((n,3))
            for i in range(N_wp):
                d_ordered[i*self.N_DER,:] = points[i,:3]
            return np.inf, d_ordered
#             return np.finfo('d').max, d_ordered
        
        d_ordered = np.zeros((n,3))
        d_ordered[:,0] = x.value[:n]
        d_ordered[:,1] = x.value[n:2*n]
        d_ordered[:,2] = x.value[2*n:]
        
        d_ordered_ret = self.get_alpha_matrix(1/t_set_scale,N_wp).dot(d_ordered)
        d_ordered_all = np.vstack([d_ordered_ret[:,0:1],d_ordered_ret[:,1:2],d_ordered_ret[:,2:3]])
        
        Q_pos = self.generate_sum_matrix(t_set, der=4, flag_loop=flag_loop)    
        Q_obj = scipy.linalg.block_diag(Q_pos,Q_pos,Q_pos)
        res = d_ordered_all.T.dot(Q_obj).dot(d_ordered_all)
        
        return res, d_ordered_ret
    
    def acc_obj(self, t_set, b, b_ext_init=None, b_ext_end=None, 
                 deg_init_min=0, deg_init_max=2, deg_end_min=0, deg_end_max=0):
        flag_loop = self.check_flag_loop(t_set,b)
        N_POLY = t_set.shape[0]
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
            P = self.generate_perm_matrix(t_set.shape[0]-1, self.N_DER_YAW)
        else:
            P = self.generate_perm_matrix(t_set.shape[0], self.N_DER_YAW)
        
        Q_yaw = self.generate_sum_matrix_yaw(t_set, der=2, flag_loop=flag_loop)
        R = P.dot(Q_yaw).dot(P.T)
        
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
        res = np.trace(d_tmp.T.dot(R).dot(d_tmp))
        d_ordered = P.T.dot(d_tmp)
        
        if res < -1e-3:
            res = 1e10
        return res, d_ordered
    
    def snap_acc_obj(self, t_set, points, plane_pos_set, \
                     deg_init_min=0, deg_init_max=4, deg_end_min=0, deg_end_max=0, \
                     deg_init_yaw_min=0, deg_init_yaw_max=4, deg_end_yaw_min=0, deg_end_yaw_max=0, \
                     flag_init_point=True, flag_fixed_point=False, \
                     flag_fixed_end_point=False, \
                     yaw_mode=0, kt=0, mu=1.0):
        
        pos_obj = lambda x: self.snap_obj( \
             x, points, plane_pos_set, \
             deg_init_min=deg_init_min, deg_init_max=deg_init_max, \
             deg_end_min=deg_end_min, deg_end_max=deg_end_max, \
             flag_fixed_end_point=flag_fixed_end_point,\
             flag_init_point=flag_init_point, flag_fixed_point=flag_fixed_point)
        
        yaw_obj = lambda x, b: self.acc_obj( \
             x, b, \
             deg_init_min=deg_init_yaw_min, deg_init_max=deg_init_yaw_max, \
             deg_end_min=deg_end_yaw_min, deg_end_max=deg_end_yaw_max)
        
        N_wp = points.shape[0]
        if kt == 0:
            t_set_scale = 10.0*t_set.shape[0]/np.sum(t_set)
        else:
            t_set_scale = 1.0
        x_t = t_set*t_set_scale
        res, d_ordered = pos_obj(x_t)
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
        res_yaw, d_ordered_yaw = yaw_obj(x=x_t, b=b_yaw)
        res += mu*res_yaw + kt*np.sum(x_t)
        
        d_ordered_ret = self.get_alpha_matrix(1/t_set_scale,N_wp).dot(d_ordered)
        d_ordered_yaw_ret = self.get_alpha_matrix_yaw(1/t_set_scale,N_wp).dot(d_ordered_yaw)
        return res, d_ordered_ret, d_ordered_yaw_ret
    
    ###############################################################################
    def get_min_snap_traj(self, points, plane_pos_set, alpha_scale=1.0, \
                          flag_loop=False, yaw_mode=0, \
                          deg_init_min=0, deg_init_max=4, \
                          deg_end_min=0, deg_end_max=0, \
                          deg_init_yaw_min=0, deg_init_yaw_max=2, \
                          deg_end_yaw_min=0, deg_end_yaw_max=0, \
                          t_set_init=None,
                          mu=1.0, kt=0, flag_fixed_end_point=False, \
                          flag_rand_init=False, flag_numpy_opt=False, flag_scipy_opt=True):
        pos_yaw_obj = lambda x: self.snap_acc_obj(
            t_set=x, points=points, plane_pos_set=plane_pos_set, \
            deg_init_min=deg_init_min, deg_init_max=deg_init_max, \
            deg_end_min=deg_end_min, deg_end_max=deg_end_max, \
            deg_init_yaw_min=deg_init_yaw_min, deg_init_yaw_max=deg_init_yaw_max, \
            deg_end_yaw_min=deg_end_yaw_min, deg_end_yaw_max=deg_end_yaw_max, \
            flag_init_point=True, flag_fixed_point=False, \
            flag_fixed_end_point=flag_fixed_end_point, \
            yaw_mode=yaw_mode, kt=kt, mu=mu)
        
        def f_obj(x):
            res, d_ordered, d_ordered_yaw = pos_yaw_obj(x)
            return res
        
        if flag_loop:
            N_POLY = len(plane_pos_set)
        else:
            N_POLY = len(plane_pos_set)-1
        
        if np.all(t_set_init == None):
            t_set = np.linalg.norm(np.diff(points[:,:3], axis=0),axis=1)*2
            if flag_loop:
                t_set = np.append(t_set,np.linalg.norm(points[-1,:3]-points[0,:3])*2)
            t_set_scale = 10.0*t_set.shape[0]/np.sum(t_set)
            t_set *= t_set_scale
        else:
            t_set = t_set_init
                
        MAX_ITER = 500
        lr = 0.5*N_POLY
        dt = 1e-3
        
        print("t_set_i: {}".format(t_set))
        
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
            print("t_set_rand_init: {}".format(t_set))
        
        if flag_numpy_opt:
            # Optimizae time ratio
            for t in range(MAX_ITER):
                grad = np.zeros_like(t_set)
                f0 = f_obj(t_set)
                for i in range(t_set.shape[0]):
                    t_set_tmp = copy.deepcopy(t_set)
                    t_set_tmp[i] += dt
                    f1 = f_obj(t_set_tmp)
                    grad[i] = (f1-f0)/dt

                err = np.mean(np.abs(grad))
                grad /= np.linalg.norm(grad)

                t_set_tmp = t_set-lr*grad

                if np.any(t_set_tmp < 0.0):
                    lr *= 0.1
                    continue

                f_tmp = f_obj(t_set_tmp)
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
        
        if flag_scipy_opt:
            bounds = []
            for i in range(t_set.shape[0]):
                bounds.append((0.01, 100.0))

            res_x, res_f, res_d = scipy.optimize.fmin_l_bfgs_b(\
                                        f_obj, x0=t_set, bounds=bounds, \
                                        approx_grad=True, epsilon=1e-4, maxiter=MAX_ITER, \
                                        iprint=1)
            t_set = np.array(res_x)
        
        rel_snap, d_ordered, d_ordered_yaw = pos_yaw_obj(t_set)
        print("t_set_snap_opt: {}".format(t_set))
        print("Relative snap: {}".format(rel_snap))

        points_new = np.zeros_like(points)
        for i in range(points.shape[0]):
            points_new[i,:3] = d_ordered[i*self.N_DER,:]
        
        return self.optimize_alpha(points_new, t_set, d_ordered, d_ordered_yaw, alpha_scale)
    
    def update_traj(self, t_set, points, plane_pos_set, alpha_set=None, \
                    yaw_mode=0, flag_run_sim=False, \
                    flag_fixed_end_point=True, \
                    flag_fixed_point=False, flag_return_snap=False):
        
        flag_update_points = False        
        if np.any(alpha_set==None):
            alpha_set=1.0*np.ones_like(t_set)
        
        if alpha_set.shape[0] > t_set.shape[0] and flag_fixed_point:
            flag_update_points = True
        
        pos_yaw_obj = lambda x: self.snap_acc_obj(
            t_set=x, points=points, plane_pos_set=plane_pos_set, \
            deg_init_min=0,deg_init_max=4, \
            deg_end_min=0,deg_end_max=2, \
            deg_init_yaw_min=0,deg_init_yaw_max=4, \
            deg_end_yaw_min=0,deg_end_yaw_max=2, \
            flag_fixed_end_point=flag_fixed_end_point, \
            flag_init_point=True, flag_fixed_point=flag_fixed_point, \
            yaw_mode=yaw_mode)
        
        t_set_new = np.multiply(t_set, alpha_set)
        res, d_ordered, d_ordered_yaw = pos_yaw_obj(t_set_new)
        
        if flag_run_sim:
            debug_array = self.sim.run_simulation_from_der( \
                t_set=t_set_new, d_ordered=d_ordered, d_ordered_yaw=d_ordered_yaw, \
                max_pos_err=0.2, min_pos_err=0.1, freq_ctrl=200, freq_sim=400)
            self.sim.plot_result(debug_array[0], save_dir="../trajectory/result", save_idx="0", t_set=t_set_new, d_ordered=d_ordered)
        
        if flag_return_snap:
            pos_yaw_obj2 = lambda x: self.snap_acc_obj(
                t_set=x, points=points, plane_pos_set=plane_pos_set, \
                deg_init_min=0,deg_init_max=4, \
                deg_end_min=0,deg_end_max=4, \
                deg_init_yaw_min=0,deg_init_yaw_max=2, \
                deg_end_yaw_min=0,deg_end_yaw_max=2, \
                flag_fixed_end_point=True, \
                flag_init_point=True, flag_fixed_point=False, \
                yaw_mode=yaw_mode)

            res_i, _, _ = pos_yaw_obj2(t_set)
            t_set_scaled = np.multiply(t_set, alpha_set)
            t_set_scaled *= np.sum(t_set)/np.sum(t_set_scaled)
            res_f, _, _ = pos_yaw_obj2(t_set_scaled)
            return t_set_new, d_ordered, d_ordered_yaw, res_f/res_i
        else:
            return t_set_new, d_ordered, d_ordered_yaw
    
    def wrapper_sanity_check(self, args):
        points = args[0]
        plane_pos_set = args[1]
        t_set = args[2]
        alpha_set = args[3]
        flag_fixed_point = args[4]
        
        t_set_new, d_ordered, d_ordered_yaw = \
            self.update_traj(t_set, points, plane_pos_set, alpha_set=alpha_set, \
                yaw_mode=self.yaw_mode, flag_run_sim=False, \
                flag_fixed_end_point=True, \
                flag_fixed_point=flag_fixed_point)
        
        return self.sanity_check(t_set_new, d_ordered, d_ordered_yaw)
    
    # run simulation with multiple loops & rampin
    def run_sim_loop(self, t_set, d_ordered, d_ordered_yaw, plane_pos_set, \
                     N_loop=1, flag_debug=False, max_pos_err=2.0, max_col_err=0.1, N_trial=3):        
        flag_loop = False
        N_wp = np.int(d_ordered.shape[0]/self.N_DER)
        N_POLY = t_set.shape[0]
        if N_wp == N_POLY:
            flag_loop = True
        
        t_set_new = t_set
        N_success_t = 0
        for idx_trial_t in range(N_trial):
            debug_array = self.sim.run_simulation_from_der(t_set_new, d_ordered, d_ordered_yaw, N_trial=1, 
                                                           max_pos_err=max_pos_err, min_pos_err=0.1, 
                                                           max_yaw_err=120., min_yaw_err=60., 
                                                           freq_ctrl=200)

            time_array = debug_array[0]["time"]
            pos_array = debug_array[0]["pos"]
            p_ii = 0
            p_ii_n = 1
            t_bias = 0
            flag_update_poly = True
            for idx in range(time_array.shape[0]):
                if (time_array[idx]-t_bias) > t_set_new[p_ii]:
                    t_bias += t_set_new[p_ii]
                    p_ii += 1
                    if flag_loop:
                        p_ii_n = (p_ii+1)%(N_POLY)
                    else:
                        p_ii_n = (p_ii+1)
                    if p_ii == N_POLY:
                        break
                    flag_update_poly = True
                elif idx == 0:
                    flag_update_poly = True
                else:
                    flag_update_poly = False

                if flag_update_poly:
                    V_norm = np.zeros((len(plane_pos_set[p_ii]["constraints_plane"]),3))
                    V_bias = np.zeros(len(plane_pos_set[p_ii]["constraints_plane"]))
                    for i in range(len(plane_pos_set[p_ii]["constraints_plane"])):
                        v0 = np.array(plane_pos_set[p_ii]["constraints_plane"][i][0])
                        v1 = np.array(plane_pos_set[p_ii]["constraints_plane"][i][1])
                        for k in range(2,len(plane_pos_set[p_ii]["constraints_plane"][i])):
                            v2 = np.array(plane_pos_set[p_ii]["constraints_plane"][i][k])
                            res_t = np.fabs((v1-v0).dot(v2-v0)/np.linalg.norm(v1-v0)/np.linalg.norm(v2-v0))
                            if res_t < 1-1e-4:
                                break
                        V_norm[i,:] = np.cross(v1-v0, v2-v0)
                        V_norm[i,:] /= np.linalg.norm(V_norm[i,:])
                        V_bias[i] = V_norm[i,:].dot(v0)

                if np.any(V_norm.dot(pos_array[idx,:])-V_bias < -max_col_err):
                    if debug_array[0]["failure_idx"] == -1:
                        debug_array[0]["failure_idx"] = idx
                    else:
                        failure_idx = debug_array[0]["failure_idx"]
                        if failure_idx > idx:
                            debug_array[0]["failure_idx"] = idx
                    prRed("crashed at {}".format(idx))
                    break

            if flag_debug:
                self.plot_sim_result(t_set_new, d_ordered, plane_pos_set, debug_array[0])

            failure_idx = debug_array[0]["failure_idx"]
            if failure_idx != -1:
                return False
        return True
        
    def plot_sim_result(self, t_set, d_ordered, plane_pos_set, debug_value, flag_course_loop=True):
        N_POLY = t_set.shape[0]
        N_wp = np.int(d_ordered.shape[0]/self.N_DER)
        flag_loop = True
        if N_POLY != N_wp:
            flag_loop = False
        
        V_t = self.generate_sampling_matrix(t_set, N=self.N_POINTS, der=0, endpoint=True)
        status = V_t.dot(d_ordered)
        # Plot data
        mesh_data = []
        if flag_loop:
            N_mesh = N_POLY
        elif flag_course_loop:
            N_mesh = N_wp
        else:
            N_mesh = N_POLY
        
        for i in range(N_mesh):
            if flag_loop or flag_course_loop:
                i_n = (i+1)%N_mesh
            else:
                i_n = (i+1)
            mesh_t = go.Mesh3d(
                # 8 vertices of a cube
                x=plane_pos_set[i][0,:].tolist()+plane_pos_set[i_n][0,:].tolist(),
                y=plane_pos_set[i][1,:].tolist()+plane_pos_set[i_n][1,:].tolist(),
                z=plane_pos_set[i][2,:].tolist()+plane_pos_set[i_n][2,:].tolist(),
                color='lightpink', opacity=0.50,
                showscale=True,
                i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            )
            mesh_data.append(mesh_t)
            
            if i == 0:
                mesh_color = 'green'
                mesh_opacity = 0.6
            else:
                mesh_color = 'orange'
                mesh_opacity = 1.0
            
            mesh_t = go.Mesh3d(
                # 8 vertices of a cube
                x=plane_pos_set[i][0,:].tolist(),
                y=plane_pos_set[i][1,:].tolist(),
                z=plane_pos_set[i][2,:].tolist(),
                color=mesh_color, opacity=mesh_opacity,
                showscale=True,
                i = [0, 0],
                j = [1, 2],
                k = [2, 3],
            )
            mesh_data.append(mesh_t)
        mesh_data.append(
            go.Scatter3d(
                x=status[:,0], 
                y=status[:,1], 
                z=status[:,2],
                mode='lines',
                name="reference",
                line=dict(
                    color='darkblue',
                    width=3,
                    dash='dash'
                )
            )
        )
        failure_idx = debug_value["failure_idx"]
        if failure_idx != -1:
            mesh_data.append(
                go.Scatter3d(
                    x=debug_value["pos"][:failure_idx,0], 
                    y=debug_value["pos"][:failure_idx,1], 
                    z=debug_value["pos"][:failure_idx,2],
                    mode='lines',
                    name="simulation",
                    line=dict(
                        color='green',
                        width=3
                    )
                )
            )
            mesh_data.append(
                go.Scatter3d(
                    x=debug_value["pos"][failure_idx:failure_idx+1,0], 
                    y=debug_value["pos"][failure_idx:failure_idx+1,1], 
                    z=debug_value["pos"][failure_idx:failure_idx+1,2],
                    mode='markers',
                    name="crash_point",
                    marker=dict(
                        size=5,
                        color='red',
                    ),
                )
            )
        else:
            mesh_data.append(
                go.Scatter3d(
                    x=debug_value["pos"][:,0], 
                    y=debug_value["pos"][:,1], 
                    z=debug_value["pos"][:,2],
                    mode='lines',
                    name="simulation",
                    line=dict(
                        color='green',
                        width=3
                    )
                )
            )
            
        waypoints = status[0:1,:]
        for i in range(1,N_POLY):
            waypoints = np.append(waypoints,status[i*self.N_POINTS:i*self.N_POINTS+1,:],axis=0)
        waypoints = np.append(waypoints,status[-1:,:],axis=0)

        mesh_data.append(
            go.Scatter3d(
                x=waypoints[:,0], 
                y=waypoints[:,1], 
                z=waypoints[:,2],
                mode='markers',
                name="waypoints",
                marker=dict(
                    size=4,
                    color='green',
                ),
            )
        )
        fig = go.Figure(data=mesh_data)
        fig.update_layout(scene_aspectmode='data')
        fig.show()

    def plot_mfbo_trajectory(self, t_set, d_ordered, plane_pos_set, t_set_new, d_ordered_new, flag_course_loop=True):
        N_POLY = t_set.shape[0]
        N_wp = np.int(d_ordered.shape[0]/self.N_DER)
        flag_loop = True
        if N_POLY != N_wp:
            flag_loop = False
        
        # Plot data
        mesh_data = []
        if flag_loop:
            N_mesh = N_POLY
        elif flag_course_loop:
            N_mesh = N_wp
        else:
            N_mesh = N_POLY
        
        for i in range(N_mesh):
            if flag_loop or flag_course_loop:
                i_n = (i+1)%N_mesh
            else:
                i_n = (i+1)
            mesh_t = go.Mesh3d(
                # 8 vertices of a cube
                x=plane_pos_set[i][0,:].tolist()+plane_pos_set[i_n][0,:].tolist(),
                y=plane_pos_set[i][1,:].tolist()+plane_pos_set[i_n][1,:].tolist(),
                z=plane_pos_set[i][2,:].tolist()+plane_pos_set[i_n][2,:].tolist(),
                color='lightpink', opacity=0.50,
                showscale=True,
                i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            )
            mesh_data.append(mesh_t)
            
            if i == 0:
                mesh_color = 'green'
                mesh_opacity = 0.6
            else:
                mesh_color = 'orange'
                mesh_opacity = 1.0
            
            mesh_t = go.Mesh3d(
                # 8 vertices of a cube
                x=plane_pos_set[i][0,:].tolist(),
                y=plane_pos_set[i][1,:].tolist(),
                z=plane_pos_set[i][2,:].tolist(),
                color=mesh_color, opacity=mesh_opacity,
                showscale=True,
                i = [0, 0],
                j = [1, 2],
                k = [2, 3],
            )
            mesh_data.append(mesh_t)
        
        V_t = self.generate_sampling_matrix(t_set, N=self.N_POINTS, der=0, endpoint=True)
        status = V_t.dot(d_ordered)
        mesh_data.append(
            go.Scatter3d(
                x=status[:,0], 
                y=status[:,1], 
                z=status[:,2],
                mode='lines',
                name="initial trajectory",
                line=dict(
                    color='darkblue',
                    width=3,
                    dash='dash'
                )
            )
        )
        
        waypoints = status[0:1,:]
        for i in range(1,N_POLY):
            waypoints = np.append(waypoints,status[i*self.N_POINTS:i*self.N_POINTS+1,:],axis=0)
        waypoints = np.append(waypoints,status[-1:,:],axis=0)
        mesh_data.append(
            go.Scatter3d(
                x=waypoints[:,0], 
                y=waypoints[:,1], 
                z=waypoints[:,2],
                mode='markers',
                name="initial waypoints",
                marker=dict(
                    size=4,
                    color='green',
                ),
            )
        )
        
        V_t = self.generate_sampling_matrix(t_set_new, N=self.N_POINTS, der=0, endpoint=True)
        status_new = V_t.dot(d_ordered_new)
        mesh_data.append(
            go.Scatter3d(
                x=status_new[:,0], 
                y=status_new[:,1], 
                z=status_new[:,2],
                mode='lines',
                name="final trajectory",
                line=dict(
                    color='orangered',
                    width=3,
                )
            )
        )
            
        waypoints = status_new[0:1,:]
        for i in range(1,N_POLY):
            waypoints = np.append(waypoints,status_new[i*self.N_POINTS:i*self.N_POINTS+1,:],axis=0)
        waypoints = np.append(waypoints,status_new[-1:,:],axis=0)
        mesh_data.append(
            go.Scatter3d(
                x=waypoints[:,0], 
                y=waypoints[:,1], 
                z=waypoints[:,2],
                mode='markers',
                name="final waypoints",
                marker=dict(
                    size=4,
                    color='crimson',
                ),
            )
        )
        
        fig = go.Figure(data=mesh_data)
        
        camera = dict(
            eye=dict(x=0.0, y=-1.35, z=2)
        )

        fig.update_layout(height=600, width=800, scene_aspectmode='data', \
            scene_camera=camera,
            legend=dict(
                x=0.6,
                y=1,
                traceorder="normal",
                bgcolor="white",),
            scene = dict(
                xaxis_title='',
                yaxis_title='',
                zaxis_title='',
                zaxis = dict(
                    tickmode = 'linear',
                    tick0 = 0.,
                    dtick = 1.0
                )
            ),
        )
        fig.show()
    
    def get_plot_points(self, t_set, d_ordered):
        N_POLY = t_set.shape[0]
        N_wp = np.int(d_ordered.shape[0]/self.N_DER)
        flag_loop = True
        if N_POLY != N_wp:
            flag_loop = False
        V_t = self.generate_sampling_matrix(t_set, N=self.N_POINTS, der=0, endpoint=True)
        status = V_t.dot(d_ordered)
        waypoints = np.zeros((N_wp,3))
        for i in range(N_wp):
            waypoints[i,:3] = d_ordered[i*self.N_DER,:]
        return status, waypoints
        
    def get_waypoints(self, t_set, d_ordered, d_ordered_yaw):
        N_wp = np.int(d_ordered.shape[0]/self.N_DER)
        waypoints = np.zeros((N_wp,4))
        for i in range(N_wp):
            waypoints[i,:3] = d_ordered[i*self.N_DER,:]
            waypoints[i,3] = np.arctan2(d_ordered_yaw[i*self.N_DER_YAW,0], d_ordered_yaw[i*self.N_DER_YAW,1])
        return waypoints
   
    def plot_trajectory(self, t_set, d_ordered, plane_pos_set, flag_course_loop=True):
        N_POLY = t_set.shape[0]
        N_wp = np.int(d_ordered.shape[0]/self.N_DER)
        flag_loop = True
        if N_POLY != N_wp:
            flag_loop = False
        
        V_t = self.generate_sampling_matrix(t_set, N=self.N_POINTS, der=0, endpoint=True)
        status = V_t.dot(d_ordered)
        # Plot data
        mesh_data = []
        if flag_loop:
            N_mesh = N_POLY
        elif flag_course_loop:
            N_mesh = N_wp
        else:
            N_mesh = N_POLY
        
        for i in range(len(plane_pos_set)):
            for j in range(len(plane_pos_set[i]["constraints_plane"])):
                p_t = np.array(plane_pos_set[i]["constraints_plane"][j])
                ijk_t = np.array([[0,k+1,k+2] for k in range(p_t.shape[0]-2)])
                mesh_t = go.Mesh3d(
                    x=list(p_t[:,0]), y=list(p_t[:,1]), z=list(p_t[:,2]),
                    color='lightpink', opacity=0.20, showscale=True,
                    i=ijk_t[:,0], j=ijk_t[:,1], k=ijk_t[:,2],)
                mesh_data.append(mesh_t)
            if len(plane_pos_set[i]["input_plane"]) > 0:
                p_t = np.array(plane_pos_set[i]["input_plane"])
                ijk_t = np.array([[0,k+1,k+2] for k in range(p_t.shape[0]-2)])
                mesh_t = go.Mesh3d(
                    x=list(p_t[:,0]), y=list(p_t[:,1]), z=list(p_t[:,2]),
                    color='green', opacity=0.20, showscale=True,
                    i=ijk_t[:,0], j=ijk_t[:,1], k=ijk_t[:,2],)
                mesh_data.append(mesh_t)
            if len(plane_pos_set[i]["output_plane"]) > 0:
                p_t = np.array(plane_pos_set[i]["output_plane"])
                ijk_t = np.array([[0,k+1,k+2] for k in range(p_t.shape[0]-2)])
                mesh_t = go.Mesh3d(
                    x=list(p_t[:,0]), y=list(p_t[:,1]), z=list(p_t[:,2]),
                    color='blue', opacity=0.20, showscale=True,
                    i=ijk_t[:,0], j=ijk_t[:,1], k=ijk_t[:,2],)
                mesh_data.append(mesh_t)
            for j in range(len(plane_pos_set[i]["corner_plane"])):
                p_t = np.array(plane_pos_set[i]["corner_plane"][j])
                ijk_t = np.array([[0,k+1,k+2] for k in range(p_t.shape[0]-2)])
                mesh_t = go.Mesh3d(
                    x=list(p_t[:,0]), y=list(p_t[:,1]), z=list(p_t[:,2]),
                    color='orange', opacity=0.40, showscale=True,
                    i=ijk_t[:,0], j=ijk_t[:,1], k=ijk_t[:,2],)
                mesh_data.append(mesh_t)
        
        mesh_data.append(
            go.Scatter3d(
                x=status[:,0], 
                y=status[:,1], 
                z=status[:,2],
                mode='lines',
                line=dict(
                    color='darkblue',
                    width=2
                )
            )
        )
        waypoints = status[0:1,:]
        for i in range(1,N_POLY):
            waypoints = np.append(waypoints,status[i*self.N_POINTS:i*self.N_POINTS+1,:],axis=0)
        waypoints = np.append(waypoints,status[-1:,:],axis=0)

        mesh_data.append(
            go.Scatter3d(
                x=waypoints[:,0], 
                y=waypoints[:,1], 
                z=waypoints[:,2],
                mode='markers',
                marker=dict(
                    size=4,
                    color='green',
                ),
            )
        )
        fig = go.Figure(data=mesh_data)
        fig.update_layout(scene_aspectmode='data')
        fig.show()
    
    def der_to_point(self, d_ordered, flag_print=False):
        N_points = np.int(d_ordered.shape[0]/self.N_DER)
        points = np.zeros((N_points,4))
        for i in range(N_points):
            points[i,:3] = d_ordered[i*self.N_DER,:]
         
        if flag_print:
            for i in range(points.shape[0]):
                print("- [{}]".format(','.join([str(x) for x in points[i,:]])))

        return points
        
    ###############################################################################
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
                    for j in range(1,self.N_POINTS-1):
                        vel_tmp = V_t.dot(d_ordered[:,:2])[i*self.N_POINTS+j,:]
                        if np.linalg.norm(vel_tmp[:2]) > 1e-6:
                            vel_tmp2 = vel_tmp
                            break
                        vel_tmp2 = np.array([1,0])
                else:
                    for j in range(1,self.N_POINTS-1):
                        vel_tmp = V_t.dot(d_ordered[:,:2])[i*self.N_POINTS-j,:]
                        if np.linalg.norm(vel_tmp[:2]) > 1e-6:
                            vel_tmp2 = vel_tmp
                            break
                        vel_tmp2 = np.array([1,0])
                yaw_ref[i,0] = -vel_tmp2[0]/np.linalg.norm(vel_tmp2[:2])
                yaw_ref[i,1] = -vel_tmp2[1]/np.linalg.norm(vel_tmp2[:2])
            else:
                yaw_ref[i,0] = -vel[i,0]/np.linalg.norm(vel[i,:2])
                yaw_ref[i,1] = -vel[i,1]/np.linalg.norm(vel[i,:2])

        if np.any(np.abs(np.linalg.norm(yaw_ref, axis=1)-1)>1e-3):
            print("Wrong yaw forward")
            sys.exit(0)
            
        return yaw_ref
    
    def get_alpha_matrix(self, alpha, N_wp):
        T_alpha = np.diag(self.generate_basis(1./alpha,self.N_DER-1,0))
        T_alpha_all = np.zeros((self.N_DER*N_wp,self.N_DER*N_wp))
        for i in range(N_wp):
            T_alpha_all[i*self.N_DER:(i+1)*self.N_DER,i*self.N_DER:(i+1)*self.N_DER] = T_alpha
        return T_alpha_all

    def get_alpha_matrix_yaw(self, alpha, N_wp):
        T_alpha = np.diag(self.generate_basis(1./alpha,self.N_DER_YAW-1,0))
        T_alpha_all = np.zeros((self.N_DER_YAW*N_wp,self.N_DER_YAW*N_wp))
        for i in range(N_wp):
            T_alpha_all[i*self.N_DER_YAW:(i+1)*self.N_DER_YAW,i*self.N_DER_YAW:(i+1)*self.N_DER_YAW] = T_alpha
        return T_alpha_all
    
    def optimize_alpha(self, points, t_set, d_ordered, d_ordered_yaw, alpha_scale=1.0, sanity_check_t=None, flag_return_alpha=False):
        if sanity_check_t == None:
            sanity_check_t = self.sanity_check

        # Optimizae alpha
        alpha = 2.0
        dalpha = 0.1
        alpha_tmp = alpha
        t_set_ret = copy.deepcopy(t_set)
        d_ordered_ret = copy.deepcopy(d_ordered)
        N_wp = np.int(d_ordered.shape[0]/self.N_DER)
        
        if np.all(d_ordered_yaw != None):
            d_ordered_yaw_ret = copy.deepcopy(d_ordered_yaw)
        else:
            d_ordered_yaw_ret = None
        
        while True:
            t_set_opt = t_set * alpha
            d_ordered_opt = self.get_alpha_matrix(alpha,N_wp).dot(d_ordered)
            if np.all(d_ordered_yaw != None):
                d_ordered_yaw_opt = self.get_alpha_matrix_yaw(alpha,N_wp).dot(d_ordered_yaw)
            else:
                d_ordered_yaw_opt = None
            
            if not sanity_check_t(t_set_opt, d_ordered_opt, d_ordered_yaw_opt):
                alpha += 1.0
            else:
                break
            
        while True:
            alpha_tmp = alpha - dalpha
            t_set_opt = t_set * alpha_tmp
            d_ordered_opt = self.get_alpha_matrix(alpha_tmp,N_wp).dot(d_ordered)
            if np.all(d_ordered_yaw != None):
                d_ordered_yaw_opt = self.get_alpha_matrix_yaw(alpha_tmp,N_wp).dot(d_ordered_yaw)
            else:
                d_ordered_yaw_opt = None
            
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
        d_ordered = self.get_alpha_matrix(alpha_scale,N_wp).dot(d_ordered_ret)
        if np.all(d_ordered_yaw != None):
            d_ordered_yaw = self.get_alpha_matrix_yaw(alpha_scale,N_wp).dot(d_ordered_yaw_ret)
        else:
            d_ordered_yaw = None
        
        if flag_return_alpha:
            return t_set, d_ordered, d_ordered_yaw, alpha
        else:
            return t_set, d_ordered, d_ordered_yaw

if __name__ == "__main__":
    poly = MinSnapTraj(MAX_POLY_DEG = 9, MAX_SYS_DEG = 4, N_POINTS = 40)
    
    

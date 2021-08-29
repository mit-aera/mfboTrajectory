#!/usr/bin/env python
# coding: utf-8

import os, sys, time, copy
import numpy as np
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os import path
import argparse
import torch

from pyTrajectoryUtils.pyTrajectoryUtils.utils import *
from mfboTrajectory.utils import *
from mfboTrajectory.agents import ActiveMFDGP
from mfboTrajectory.minSnapTrajectoryWaypoints import MinSnapTrajectoryWaypoints
from mfboTrajectory.multiFidelityModelWaypoints import meta_low_fidelity, meta_high_fidelity, get_dataset_init, check_dataset_init

if __name__ == "__main__":
    yaml_name = 'constraints_data/waypoints_constraints.yaml'
    sample_name = ['traj_1', 'traj_2', 'traj_3', 'traj_4', 'traj_5', 'traj_6', 'traj_7', 'traj_8']
    drone_model = "default"
    
    rand_seed = [123, 445, 678, 115, 92, 384, 992, 874, 490, 41, 83, 78, 991, 993, 994, 995, 996, 997, 998, 999]
    MAX_ITER = 5

    parser = argparse.ArgumentParser(description='mfbo experiment')
    parser.add_argument('-l', dest='flag_load_exp_data', action='store_true', help='load exp data')
    parser.add_argument('-g', dest='flag_switch_gpu', action='store_true', help='switch gpu to gpu 1')
    parser.add_argument('-t', "--sample_idx", type=int, help="assign model index", default=0)
    parser.add_argument("-s", "--seed_idx", type=int, help="assign seed index", default=0)
    parser.add_argument("-y", "--yaw_mode", type=int, help="assign seed index", default=2)
    parser.add_argument("-m", "--max_iter", type=int, help="assign maximum iteration", default=50)
    parser.set_defaults(flag_load_exp_data=False)
    parser.set_defaults(flag_switch_gpu=False)
    args = parser.parse_args()

    if args.flag_switch_gpu:
        torch.cuda.set_device(1)
    else:
        torch.cuda.set_device(0)

    yaw_mode = args.yaw_mode
    sample_name_ = sample_name[args.sample_idx]
    rand_seed_ = rand_seed[args.seed_idx]
    MAX_ITER = np.int(args.max_iter)
    print("MAX_ITER: {}".format(MAX_ITER))
    
    lb = 0.1
    ub = 1.9
    if yaw_mode == 0:
        flag_yaw_zero = True
    else:
        flag_yaw_zero = False
    
    print("Trajectory {}".format(sample_name_))
    
    if yaw_mode == 0:
        sample_name_ += "_yaw_zero"
    
    print("Yaw_mode {}".format(yaw_mode))
    poly = MinSnapTrajectoryWaypoints(drone_model=drone_model, yaw_mode=yaw_mode)
    points, t_set_sta = get_waypoints(yaml_name, sample_name_, flag_t_set=True)
    t_dim = t_set_sta.shape[0]
    lb_i = np.ones(t_dim)*lb
    ub_i = np.ones(t_dim)*ub
    res_init, data_init = check_dataset_init(sample_name_, t_dim, N_L=1000, N_H=20, lb=lb, ub=ub, sampling_mode=2)
    if res_init:
        alpha_sim, X_L, Y_L, X_H, Y_H = data_init
        t_set_sim = t_set_sta * alpha_sim
        low_fidelity = lambda x, debug=True: meta_low_fidelity(poly, x, t_set_sta, points, debug, lb=lb, ub=ub)
        high_fidelity = lambda x, return_snap=False: meta_high_fidelity(poly, x, t_set_sim, points, lb=lb, ub=ub, return_snap=return_snap)
    else:
        print("Initializing dataset")
        sanity_check_t = lambda t_set, d_ordered, d_ordered_yaw: \
            poly.run_sim_loop(t_set, d_ordered, d_ordered_yaw)
        t_set_sta, d_ordered, d_ordered_yaw = poly.update_traj_(points, t_set_sta, np.ones_like(t_set_sta))
        t_set_sim, d_ordered, d_ordered_yaw, alpha_sim = \
            poly.optimize_alpha(points, t_set_sta, d_ordered, d_ordered_yaw, alpha_scale=1.0, \
                                    sanity_check_t=sanity_check_t, flag_return_alpha=True)
        print("alpha_sim: {}".format(alpha_sim))
        low_fidelity = lambda x, debug=True: meta_low_fidelity(poly, x, t_set_sta, points, debug, lb=lb, ub=ub)
        high_fidelity = lambda x, return_snap=False: meta_high_fidelity(poly, x, t_set_sim, points, lb=lb, ub=ub, return_snap=return_snap)
        X_L, Y_L, X_H, Y_H = get_dataset_init(sample_name_, alpha_sim, low_fidelity, high_fidelity, \
                                              t_dim, N_L=1000, N_H=20, lb=lb, ub=ub, sampling_mode=2)
    
    print("Seed {}".format(rand_seed_))
            
    np.random.seed(rand_seed_)
    torch.manual_seed(rand_seed_)

    fileprefix = 'test_waypoints'
    filedir = './mfbo_data/{}'.format(sample_name_)
    logprefix = '{}/{}/{}'.format(sample_name_, fileprefix, rand_seed_)
    filename_res = 'result_{}_{}.yaml'.format(fileprefix, rand_seed_)
    filename_exp = 'exp_data_{}_{}.yaml'.format(fileprefix, rand_seed_)
    res_path = os.path.join(filedir, filename_res)
    print(res_path)
    flag_check = check_result_data(filedir,filename_res,MAX_ITER)
    if not flag_check:
        mfbo_model = ActiveMFDGP( \
            X_L=X_L, Y_L=Y_L, X_H=X_H, Y_H=Y_H, \
            lb_i=lb_i, ub_i=ub_i, rand_seed=rand_seed_, \
            C_L=1.0, C_H=10.0, \
            delta_L=0.9, delta_H=0.6, \
            beta=3.0, N_cand=16384, \
            gpu_batch_size=1024, \
            sampling_func_L=low_fidelity, \
            sampling_func_H=high_fidelity, \
            t_set_sim=t_set_sim, \
            utility_mode=0, sampling_mode=5, \
            model_prefix=logprefix, \
            iter_create_model=200)

        path_exp_data = os.path.join(filedir, filename_exp)
        if args.flag_load_exp_data and path.exists(path_exp_data):
            mfbo_model.load_exp_data(filedir=filedir, filename=filename_exp)

        mfbo_model.active_learning( \
            N=MAX_ITER, plot=False, MAX_low_fidelity=50, \
            filedir=filedir, \
            filename_result=filename_res, \
            filename_exp=filename_exp)

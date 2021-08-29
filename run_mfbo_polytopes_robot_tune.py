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
from mfboTrajectory.minSnapTrajectoryPolytopes import MinSnapTrajectoryPolytopes
from mfboTrajectory.multiFidelityModelPolytopes import get_waypoints_plane, check_dataset_init, meta_low_fidelity, meta_high_fidelity, get_dataset_init, meta_get_waypoints_alpha
from mfboTrajectory.utilsConvexDecomp import *

if __name__ == "__main__":
    try:
        import gurobipy
        model = gurobipy.Model()
    except:
        print("No gurobi license")
        sys.exit()
    
    sample_name = ['traj_9', 'traj_10', 'traj_11', 'traj_12']
    drone_model = "default"
    
    MAX_ITER = 50

    parser = argparse.ArgumentParser(description='mfbo experiment')
    parser.add_argument('-t', "--sample_idx", type=int, help="assign model index", default=0)
    parser.add_argument("-y", "--yaw_mode", type=int, help="assign seed index", default=0)
    parser.add_argument('-a', dest='alpha', type=float, help='set alpha')
    parser.add_argument("-o", "--qp_optimizer", type=str, help="select optimizer for quadratic programming", default='osqp')
    parser.set_defaults(flag_load_exp_data=False)
    parser.set_defaults(flag_switch_gpu=False)
    args = parser.parse_args()
    alpha_tmp = np.int(np.floor((args.alpha-np.floor(args.alpha))*100))
    if alpha_tmp < 10:
        alpha_str_tmp = '0' + str(alpha_tmp)
    else:
        alpha_str_tmp = str(alpha_tmp)
    alpha_str = str(np.int(args.alpha))+"p"+alpha_str_tmp
    print("Alpha: {}".format(args.alpha))
    print("Alpha Str: {}".format(alpha_str))
    
    if args.flag_switch_gpu:
        torch.cuda.set_device(1)
    else:
        torch.cuda.set_device(0)

    qp_optimizer = args.qp_optimizer.lower()
    assert qp_optimizer in ['osqp', 'gurobi','cvxopt']
    
    yaw_mode = args.yaw_mode
    sample_name_ = sample_name[args.sample_idx]
    lb = 0.1
    ub = 1.9
    if yaw_mode == 0:
        flag_yaw_zero = True
    else:
        flag_yaw_zero = False
    
    print("Trajectory {}".format(sample_name_))
    polygon_filedir = './constraints_data'
    polygon_filename = 'polytopes_constraints.yaml'
    
    poly = MinSnapTrajectoryPolytopes(drone_model=drone_model, yaw_mode=yaw_mode, qp_optimizer=qp_optimizer)
    points, plane_pos_set, t_set_sta = get_waypoints_plane(polygon_filedir, polygon_filename, sample_name_, flag_t_set=True)
    
    lb = 0.1
    ub = 1.9
    
    if yaw_mode == 0:
        sample_name_ += "_yaw_zero"
    
    print("Yaw_mode {}".format(yaw_mode))
    
    t_dim = t_set_sta.shape[0]
    lb_i = np.ones(t_dim)*lb
    ub_i = np.ones(t_dim)*ub

    t_set = t_set_sta*args.alpha
    print("t_set: {}".format(t_set))
    alpha_set = np.ones(t_dim)
    
    t_set_new, d_ordered, d_ordered_yaw = poly.update_traj(
            t_set, points, plane_pos_set, alpha_set, 
            flag_fixed_point=False, flag_fixed_end_point=True)
    
    d_ordered_t = copy.deepcopy(d_ordered)
    d_ordered_t[:,0] = d_ordered[:,1]
    d_ordered_t[:,1] = d_ordered[:,0]
    
    if not os.path.exists("./mfbo_data/robot_exp_traj/"):
        os.makedirs("./mfbo_data/robot_exp_traj/")
    poly.save_trajectory_yaml(t_set_new, d_ordered_t, d_ordered_yaw, \
                              traj_dir="./mfbo_data/robot_exp_traj/", \
                              traj_name="{}_a{}".format(sample_name_, alpha_str))

    print(np.sum(t_set_new))

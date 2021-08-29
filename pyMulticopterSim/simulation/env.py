#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import cv2
import signal
import os, sys, time, copy, argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .utils import *
from .model import *

import functools, traceback

class simulation_env():
    def __init__(self, *args, **kwargs):
        if 'cfg_dir' in kwargs:
            cfg_dir = kwargs['cfg_dir']
        else:
            curr_path = os.path.dirname(os.path.abspath(__file__))
            cfg_dir = curr_path+"/../config/"
        
        if 'cfg_filename' in kwargs:
            cfg_filename = kwargs['cfg_filename']
        else:
            cfg_filename = "SimulationClient.yaml"
        
        if 'cfg_uav' in kwargs:
            cfg_uav = kwargs['cfg_uav']
        else:
            cfg_uav = "multicopterDynamicsSim.yaml"
            
        fgc_cfg_path = os.path.join(cfg_dir, cfg_filename)
        uav_sim_cfg_path = os.path.join(cfg_dir, cfg_uav)

        self.vehicle_set = dict()
        
        with open(fgc_cfg_path, 'r') as stream:
            try:
                cfg = yaml.safe_load(stream)
                for vehicle_key in cfg['vehicle_model'].keys():
                    if cfg['vehicle_model'][vehicle_key]['type'] == "uav":
                        vehicle_tmp = MulticopterModel(
                            cfg_path=uav_sim_cfg_path,
                            id=vehicle_key,
                            init_pose=np.array(cfg['vehicle_model'][vehicle_key]['initialPose']),
                            imu_freq=cfg['vehicle_model'][vehicle_key]['imu_freq'])
                    else:
                        continue
                    self.vehicle_set[vehicle_key] = dict()
                    self.vehicle_set[vehicle_key]["type"] = cfg['vehicle_model'][vehicle_key]['type']
                    self.vehicle_set[vehicle_key]["model"] = vehicle_tmp
                    self.vehicle_set[vehicle_key]["logs"] = []
                    
                curr_path = os.path.dirname(os.path.abspath(__file__))
                
            except yaml.YAMLError as exc:
                print(exc)
        
        self.initialize_state()
        return
    
    def manual_seed(self, seed):
        for vehicle_key in self.vehicle_set.keys():
            if self.vehicle_set[vehicle_id]["type"] == "drone":
                self.vehicle_set[vehicle_id]["model"].setRandomSeed(seed, seed)
        return

    def set_state_vehicle(self, vehicle_id, **kwargs):
        return self.vehicle_set[vehicle_id]["model"].set_state(**kwargs)
    
    def get_state(self, vehicle_id):
        return self.vehicle_set[vehicle_id]["model"].get_state()

    def initialize_state(self):
        for vehicle_key in self.vehicle_set.keys():
            self.vehicle_set[vehicle_key]["model"].initialize_state()
            self.vehicle_set[vehicle_key]["logs"] = [copy.deepcopy(self.vehicle_set[vehicle_key]["model"].get_state())]
        return

    def proceed(self, vehicle_id, speed_command, steering_angle_command, duration):
        if self.vehicle_set[vehicle_id]["type"] != "car":
            return
        self.vehicle_set[vehicle_id]["model"].proceed(speed_command, steering_angle_command, duration)
        self._update_state(vehicle_id, duration)
        return

    def proceed_motor_speed(self, vehicle_id, motor_command, duration):
        if self.vehicle_set[vehicle_id]["type"] != "uav":
            return
        self.vehicle_set[vehicle_id]["model"].proceed_motor_speed(motor_command, duration)
        self._update_state(vehicle_id, duration)
        return

    def proceed_angular_rate(self, vehicle_id, angular_rate_command, thrust_command, duration):
        if self.vehicle_set[vehicle_id]["type"] != "uav":
            return
        self.vehicle_set[vehicle_id]["model"].proceed_angular_rate(angular_rate_command, thrust_command, duration)
        self._update_state(vehicle_id, duration)
        return
    
    def proceed_waypoint(self, vehicle_id, waypoint_command, duration):
        if self.vehicle_set[vehicle_id]["type"] != "uav":
            return
        self.vehicle_set[vehicle_id]["model"].proceed_waypoint(waypoint_command, duration)
        self._update_state(vehicle_id, duration)
        return

    def save_logs(self, vehicle_id=None, save_dir="data/"):
        # Save IMU and Camera data
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        vehicle_set = []
        if np.all(vehicle_id == None):
            vehicle_set = self.vehicle_set.keys()
        else:
            vehicle_set = [vehicle_id]
        
        for vehicle_id in vehicle_set:
            if self.vehicle_set[vehicle_id]["type"] == "uav":
                arr_timestamp = []
                arr_acc_raw_x = []
                arr_acc_raw_y = []
                arr_acc_raw_z = []
                arr_gyro_raw_x = []
                arr_gyro_raw_y = []
                arr_gyro_raw_z = []
                for log in self.vehicle_set[vehicle_id]["logs"]:
                    arr_timestamp.append(np.uint64(log["timestamp"]*1e9))
                    # arr_acc_raw_x.append(log["acceleration_raw"][0])
                    # arr_acc_raw_y.append(log["acceleration_raw"][1])
                    # arr_acc_raw_z.append(log["acceleration_raw"][2])
                    # arr_gyro_raw_x.append(log["gyro_raw"][0])
                    # arr_gyro_raw_y.append(log["gyro_raw"][1])
                    # arr_gyro_raw_z.append(log["gyro_raw"][2])
                    arr_acc_raw_x.append(log["acceleration"][0])
                    arr_acc_raw_y.append(log["acceleration"][1])
                    arr_acc_raw_z.append(log["acceleration"][2])
                    arr_gyro_raw_x.append(log["angular_velocity"][0])
                    arr_gyro_raw_y.append(log["angular_velocity"][1])
                    arr_gyro_raw_z.append(log["angular_velocity"][2])
                df = pd.DataFrame({
                    "timestamp":arr_timestamp,
                    "x.1":arr_gyro_raw_x,
                    "y.1":arr_gyro_raw_y,
                    "z.1":arr_gyro_raw_z,
                    "x.2":arr_acc_raw_x,
                    "y.2":arr_acc_raw_y,
                    "z.2":arr_acc_raw_z})
                df.to_csv(os.path.join(save_dir,"{}_imu.csv".format(vehicle_id)), sep=',', index=False)

                arr_timestamp = []
                arr_pos_x = []
                arr_pos_y = []
                arr_pos_z = []
                arr_vel_x = []
                arr_vel_y = []
                arr_vel_z = []
                arr_att_x = []
                arr_att_y = []
                arr_att_z = []
                for log in self.vehicle_set[vehicle_id]["logs"]:
                    arr_timestamp.append(np.uint64(log["timestamp"]*1e9))
                    arr_pos_x.append(log["position"][0])
                    arr_pos_y.append(log["position"][1])
                    arr_pos_z.append(log["position"][2])
                    arr_vel_x.append(log["velocity"][0])
                    arr_vel_y.append(log["velocity"][1])
                    arr_vel_z.append(log["velocity"][2])
                    arr_att_x.append(log["attitude_euler_angle"][0])
                    arr_att_y.append(log["attitude_euler_angle"][1])
                    arr_att_z.append(log["attitude_euler_angle"][2])
                df = pd.DataFrame({
                    "timestamp":arr_timestamp,
                    "pos.x":arr_pos_x,
                    "pos.y":arr_pos_y,
                    "pos.z":arr_pos_z,
                    "vel.x":arr_vel_x,
                    "vel.y":arr_vel_y,
                    "vel.z":arr_vel_z,
                    "att.x":arr_att_x,
                    "att.y":arr_att_y,
                    "att.z":arr_att_z})
                df.to_csv(os.path.join(save_dir,"{}_sim.csv".format(vehicle_id)), sep=',', index=False)
                
        return

    def _update_state(self, vehicle_id, duration):
        self.vehicle_set[vehicle_id]["logs"].append(copy.deepcopy(self.vehicle_set[vehicle_id]["model"].get_state()))
        return

    def plot_state(self, vehicle_id, attribute=None):
        logs = copy.deepcopy(self.vehicle_set[vehicle_id]["logs"])
        if self.vehicle_set[vehicle_id]["type"] == "uav":
            att_list = ["position", "velocity", "acceleration", "attitude_euler_angle", "angular_velocity", "angular_acceleration"]
            att_list2 = ["motor_speed", "motor_acceleration"]
            for att_ in att_list:
                if attribute == att_ or attribute == None:
                    time_array = []
                    data_array = []
                    for l in logs:
                        time_array.append(l["timestamp"])
                        data_array.append(l[att_])
                    time_array = np.array(time_array)
                    data_array = np.array(data_array)
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(time_array, data_array[:,0], '-', label='x')
                    ax.plot(time_array, data_array[:,1], '-', label='y')
                    ax.plot(time_array, data_array[:,2], '-', label='z')
                    ax.set_title("{} - {}".format(vehicle_id, att_))
                    ax.set_xlabel('time (s)')
                    ax.legend()
                    ax.grid()

            for att_ in att_list2:
                if attribute == att_ or attribute == None:
                    time_array = []
                    data_array = []
                    for l in logs:
                        time_array.append(l["timestamp"])
                        data_array.append(l[att_])
                    time_array = np.array(time_array)
                    data_array = np.array(data_array)
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(time_array, data_array[:,0], '-', label='motor_1')
                    ax.plot(time_array, data_array[:,1], '-', label='motor_2')
                    ax.plot(time_array, data_array[:,2], '-', label='motor_3')
                    ax.plot(time_array, data_array[:,3], '-', label='motor_4')
                    ax.set_title("{} - {}".format(vehicle_id, att_))
                    ax.set_xlabel('time (s)')
                    ax.legend()
                    ax.grid()
        
        return

if __name__ == "__main__":
    # execute only if run as a script
    env = flightgoggles_env()
    env.proceed_motor_speed("Vehicle_1", np.array([1100.0,1100.0,1100.0,1100.0]),0.1)
    env.plot_state()
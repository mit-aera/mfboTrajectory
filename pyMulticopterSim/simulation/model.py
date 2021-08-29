import numpy as np
import cv2
import signal
import os, sys, time, copy, argparse, yaml
from multicopter_dynamics_sim import MulticopterDynamicsSim as uav_dyns

from .controller import UAV_pid_angular_rate, UAV_pid_waypoint
from .filter import LowPassFilter
from .utils import *

class VehicleModel():
    def __init__(self, *args, **kwargs):
        if 'init_pose' in kwargs:
            self.init_pose = np.array(kwargs["init_pose"])
        else:
            self.init_pose = np.array([0,0,0,1,0,0,0])

        self.camera_pose = dict()
        self.camera_info = dict()
        if 'camera_info' in kwargs:
            self.camera_info = kwargs['camera_info']
            for cam_key in self.camera_info.keys():
                self.camera_pose[cam_key] = dict()
                self.camera_pose[cam_key]["position"] = self.init_pose[:3] + self.camera_info[cam_key]['relativePose'][:3]
                self.camera_pose[cam_key]["attitude"] = mul_quat(self.init_pose[3:7], self.camera_info[cam_key]['relativePose'][3:7])
                self.camera_pose[cam_key]["freq"] = self.camera_info[cam_key]['freq']
                self.camera_pose[cam_key]["flag_update"] = False
                self.camera_pose[cam_key]["cam_time"] = 0
        
        if 'id' in kwargs:
            self.vehicle_id = kwargs['id']

        if 'imu_freq' in kwargs:
            self.imu_freq = np.double(kwargs['imu_freq'])
        else:
            self.imu_freq = 960.0
        
        return
    
    def initialize_state(self):
        return NotImplementedError

# Dynamics coordinate: NED
# ROS coordinate: ENU
class MulticopterModel(VehicleModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'cfg_path' in kwargs:
            cfg_path = kwargs['cfg_path']
        else:
            curr_path = os.path.dirname(os.path.abspath(__file__))
            cfg_path = curr_path+"/../config/multicopterDynamicsSim.yaml"
        
        with open(cfg_path, 'r') as stream:
            try:
                cfg = yaml.safe_load(stream)
                self.gravity = 9.81

                self.uav_sim = uav_dyns(
                    numCopter=4, 
                    thrustCoefficient=np.double(cfg['flightgoggles_uav_dynamics']['thrust_coefficient']), 
                    torqueCoefficient=np.double(cfg['flightgoggles_uav_dynamics']['torque_coefficient']),
                    minMotorSpeed=0, 
                    maxMotorSpeed=np.double(cfg['flightgoggles_uav_dynamics']['max_prop_speed']),
                    motorTimeConstant=np.double(cfg['flightgoggles_uav_dynamics']['motor_time_constant']),
                    motorRotationalInertia=np.double(cfg['flightgoggles_uav_dynamics']['motor_rotational_inertia']),
                    vehicleMass=np.double(cfg['flightgoggles_uav_dynamics']['vehicle_mass']),
                    vehicleInertia=np.diag(np.array([
                        np.double(cfg['flightgoggles_uav_dynamics']['vehicle_inertia_xx']),
                        np.double(cfg['flightgoggles_uav_dynamics']['vehicle_inertia_yy']),
                        np.double(cfg['flightgoggles_uav_dynamics']['vehicle_inertia_zz'])])),
                    aeroMomentCoefficient=np.diag(np.array([
                        np.double(cfg['flightgoggles_uav_dynamics']['aeromoment_coefficient_xx']),
                        np.double(cfg['flightgoggles_uav_dynamics']['aeromoment_coefficient_yy']),
                        np.double(cfg['flightgoggles_uav_dynamics']['aeromoment_coefficient_zz'])])),
                    dragCoefficient=np.double(cfg['flightgoggles_uav_dynamics']['drag_coefficient']),
                    momentProcessNoiseAutoCorrelation=np.double(cfg['flightgoggles_uav_dynamics']['moment_process_noise']),
                    forceProcessNoiseAutoCorrelation=np.double(cfg['flightgoggles_uav_dynamics']['force_process_noise']),
                    gravity=np.array([0,0,self.gravity])
                )

                momentArm_ = np.double(cfg['flightgoggles_uav_dynamics']['moment_arm'])
                motorFrame = np.zeros((3,4))

                motorFrame[:3,:3] = np.diag(np.array([1,-1,-1]))
                motorFrame[:,3] = np.array([momentArm_,-momentArm_,0.])
                self.uav_sim.setMotorFrame(motorFrame,1,0)
                motorFrame[:,3] = np.array([momentArm_,momentArm_,0.])
                self.uav_sim.setMotorFrame(motorFrame,-1,1)
                motorFrame[:,3] = np.array([-momentArm_,momentArm_,0.])
                self.uav_sim.setMotorFrame(motorFrame,1,2)
                motorFrame[:,3] = np.array([-momentArm_,-momentArm_,0.])
                self.uav_sim.setMotorFrame(motorFrame,-1,3)
                
                # motorFrame[:3,:3] = np.diag(np.array([1,1,1]))
                # motorFrame[:,3] = np.array([momentArm_,-momentArm_,0.])
                # self.uav_sim.setMotorFrame(motorFrame,1,0)
                # motorFrame[:,3] = np.array([momentArm_,momentArm_,0.])
                # self.uav_sim.setMotorFrame(motorFrame,-1,1)
                # motorFrame[:,3] = np.array([-momentArm_,momentArm_,0.])
                # self.uav_sim.setMotorFrame(motorFrame,1,2)
                # motorFrame[:,3] = np.array([-momentArm_,-momentArm_,0.])
                # self.uav_sim.setMotorFrame(motorFrame,-1,3)

                mass = np.double(cfg['flightgoggles_uav_dynamics']['vehicle_mass'])
                thrustCoef = np.double(cfg['flightgoggles_uav_dynamics']['thrust_coefficient'])
                self.propSpeed_sta = np.sqrt(mass*self.gravity/thrustCoef/4)
                
                # self.uav_sim.setMotorSpeed(self.propSpeed_sta)
                self.uav_sim.resetMotorSpeeds()
                
                self.init_position = self.init_pose[:3]
                self.init_attitude = self.init_pose[3:7]

                # IMU
                self.uav_sim.setIMUBias(
                    np.double(cfg["flightgoggles_imu"]["accelerometer_biasinitvar"]),
                    np.double(cfg["flightgoggles_imu"]["gyroscope_biasinitvar"]),
                    np.double(cfg["flightgoggles_imu"]["accelerometer_biasprocess"]),
                    np.double(cfg["flightgoggles_imu"]["gyroscope_biasprocess"])
                )
                self.uav_sim.setIMUNoiseVariance(
                    np.double(cfg["flightgoggles_imu"]["accelerometer_variance"]),
                    np.double(cfg["flightgoggles_imu"]["gyroscope_variance"])
                )

                # Simulation Frequency
                self.sim_freq = np.double(cfg["sim_freq"])

            except yaml.YAMLError as exc:
                print(exc)

        self.controller_angrate = UAV_pid_angular_rate(vehicleMass=mass, gravity=self.gravity)
        self.controller_waypoint = UAV_pid_waypoint(vehicleMass=mass, gravity=self.gravity)

        self.lpf_acc = LowPassFilter(dim=3,
            gainP=np.double(cfg["flightgoggles_lpf"]["gain_p"]),
            gainQ=np.double(cfg["flightgoggles_lpf"]["gain_q"]))
        self.lpf_gyro = LowPassFilter(dim=3,
            gainP=np.double(cfg["flightgoggles_lpf"]["gain_p"]),
            gainQ=np.double(cfg["flightgoggles_lpf"]["gain_q"]))
        self.lpf_ms = LowPassFilter(dim=4,
            gainP=np.double(cfg["flightgoggles_lpf"]["gain_p"]),
            gainQ=np.double(cfg["flightgoggles_lpf"]["gain_q"]))
        
        self.initialize_state()
        return

    def initialize_state(self):
        self.pos = self.init_position
        self.vel = np.zeros(3)
        # self.acc = np.array([0,0,-self.gravity])
        self.acc = np.array([0,0,0])
        self.att_q = self.init_attitude
        self.att = quat2Euler(self.att_q)
        self.angV = np.zeros(3)
        self.angA = np.zeros(3)
        self.ms = np.ones(4)*self.propSpeed_sta
#         self.ms = np.zeros(4)
        self.ma = np.zeros(4)

        self.acc_raw = copy.deepcopy(self.acc)
        self.gyro_raw = np.zeros(3)
        self.ms_raw = copy.deepcopy(self.ms)

        self.controller_angrate.reset_state()
        self.controller_waypoint.reset_state()

        self.lpf_acc.reset_state(self.acc, np.zeros(3))
        self.lpf_gyro.reset_state(self.angV, self.angA)
        self.lpf_ms.reset_state(self.ms, self.ma)

        self.uav_sim.setVehicleState(
            position=self.init_position,
            velocity=np.zeros(3),
            angularVelocity=np.zeros(3),
            attitude=self.init_attitude,
            motorSpeed=np.zeros(4))

        self.sim_time = 0
        self.sim_time_dynamics = 0
        self.imu_time = 0
        for cam_key in self.camera_info.keys():
            self.camera_pose[cam_key]["position"] = self.init_pose[:3] + self.camera_info[cam_key]['relativePose'][:3]
            self.camera_pose[cam_key]["attitude"] = mul_quat(self.init_pose[3:7], self.camera_info[cam_key]['relativePose'][3:7])
            self.camera_pose[cam_key]["flag_update"] = False
            self.camera_pose[cam_key]["cam_time"] = 0
            self.camera_pose[cam_key]["cam_time_last"] = 0

        return
    
    def get_state(self):
        state = dict()
        state["timestamp"] = self.sim_time
        state["position"] = self.pos
        state["velocity"] = self.vel
        state["acceleration_raw"] = self.acc_raw
        state["acceleration"] = self.acc
        state["attitude_euler_angle"] = self.att
        state["attitude"] = self.att_q
        state["gyroscope_raw"] = self.gyro_raw
        state["angular_velocity"] = self.angV
        state["angular_acceleration"] = self.angA
        state["motor_speed_raw"] = self.ms_raw
        state["motor_speed"] = self.ms
        state["motor_acceleration"] = self.ma
        return state
    
    def set_state(self, **kwargs):        
        if "position" in kwargs:
            self.pos = kwargs["position"]
        
        if "velocity" in kwargs:
            self.vel = kwargs["velocity"]
        
        if "acceleration_raw" in kwargs:
            self.acc_raw = kwargs["acceleration_raw"]
        
        if "acceleration" in kwargs:
            self.acc = kwargs["acceleration"]
        
        if "attitude_euler_angle" in kwargs:
            self.att = kwargs["attitude_euler_angle"]
            self.att_q = Euler2quat(self.att)
        
        if "attitude" in kwargs:
            self.att_q = kwargs["attitude"]
            self.att = quat2Euler(self.att_q)
        
        if "gyroscope_raw" in kwargs:
            self.gyro_raw = kwargs["gyroscope_raw"]
        
        if "angular_velocity" in kwargs:
            self.angV = kwargs["angular_velocity"]
        
        if "angular_acceleration" in kwargs:
            self.angA = kwargs["angular_acceleration"]
        
        if "motor_speed_raw" in kwargs:
            self.ms_raw = kwargs["motor_speed_raw"]
        
        if "motor_speed" in kwargs:
            self.ms = kwargs["motor_speed"]

        if "motor_acceleration" in kwargs:
            self.ma = kwargs["motor_acceleration"]
        
        self.uav_sim.setVehicleState(self.pos, self.vel, self.angV, self.att_q, self.ms)
        uav_state_t = self.uav_sim.getVehicleState()

        self.controller_angrate.reset_state()
        self.controller_waypoint.reset_state()
        
        self.lpf_acc.reset_state(self.acc, np.zeros(3))
        self.lpf_gyro.reset_state(self.angV, self.angA)
        self.lpf_ms.reset_state(self.ms, self.ma)
        
        return
        
    def proceed_motor_speed(self, motor_command, dt):
        self.sim_time += dt
        while self.sim_time > self.sim_time_dynamics + 1./self.sim_freq:
            self.uav_sim.proceedState(1./self.sim_freq, motor_command)
            self.sim_time_dynamics += 1./self.sim_freq
            if self.sim_time_dynamics >= self.imu_time:
                self.update_state_imu(1./self.imu_freq)
                self.imu_time += 1./self.imu_freq
            self.update_state_vehicle(1./self.sim_freq)
        return

    def proceed_angular_rate(self, angular_rate_command, thrust_command, dt):
        motor_command = self.controller_angrate.control_update(angular_rate_command, thrust_command, self.angV, self.angA, dt)
        self.proceed_motor_speed(motor_command, dt)
        return
    
    def proceed_waypoint(self, waypoint_command, dt):
        motor_command = self.controller_waypoint.control_update(waypoint_command, self.pos, self.vel, self.att, self.angV, self.angA, dt)
        self.proceed_motor_speed(motor_command, dt)
        return
    
    def proceed_idle(self, dt):
        pose = np.zeros(4)
        pose[:3] = self.pos
        pose[3] = self.att[2]
        self.proceed_pose(pose, dt)
        return

    def update_state_vehicle(self, dt):
        # state from mocap and encoder
        uav_state_t = self.uav_sim.getVehicleState()
        self.pos = uav_state_t["position"]
        self.vel = uav_state_t["velocity"]
        self.att_q = uav_state_t["attitude"]
        self.att = quat2Euler(self.att_q)
        self.angV_raw = uav_state_t["angularVelocity"]
        self.ms_raw = uav_state_t["motorSpeed"]

        # print("pos: {}".format(self.pos))
        # print("vel: {}".format(self.vel))
        # self.pos, self.vel, self.angV_raw, self.att_q, self.ms_raw = self.uav_sim.getVehicleState()
        # self.att = quat2Euler(self.att_q)

        # filtered state
        self.lpf_ms.proceed_state(self.ms_raw, dt)
        self.ms = self.lpf_ms.filterState_
        self.ma = self.lpf_ms.filterStateDer_
        return

    def update_state_imu(self, dt):
        # state from imu
        imu_state_t = self.uav_sim.getIMUMeasurement()
        self.acc_raw = imu_state_t["acc"]
        self.gyro_raw = imu_state_t["gyro"]
        
        # filtered state
        self.lpf_acc.proceed_state(self.acc_raw, dt)
        self.lpf_gyro.proceed_state(self.gyro_raw, dt)
        
        self.acc = self.lpf_acc.filterState_
        self.angV = self.lpf_gyro.filterState_
        self.angA = self.lpf_gyro.filterStateDer_
        return

    def update_state_camera(self):
        for cam_key in self.camera_info.keys():
            self.camera_pose[cam_key]["position"] = self.pos + self.camera_info[cam_key]['relativePose'][:3]
            self.camera_pose[cam_key]["attitude"] = mul_quat(self.att_q, self.camera_info[cam_key]['relativePose'][3:7])
            self.camera_pose[cam_key]["flag_update"] = True
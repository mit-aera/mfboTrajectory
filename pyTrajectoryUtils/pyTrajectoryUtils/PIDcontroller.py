#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import signal
import os, sys, time, copy, argparse

from .utils import *

class LowPassFilter():
    def __init__(self, *args, **kwargs):
        if 'debug' in kwargs:
            self.flag_debug = kwargs['debug']
        else:
            self.flag_debug = False
        
        if not(('gainP' in kwargs) and ('gainQ' in kwargs)):
            if self.flag_debug:
                print("Did not get the LPF gain_p and/or gain_q from the params, defaulting to 30 Hz cutoff freq.")
            self.gainP_ = 35530.5758439217
            self.gainQ_ = 266.572976289502
        else:
            self.gainP_ = kwargs['gainP']
            self.gainQ_ = kwargs['gainQ']

        if 'dim' in kwargs:
            self.dim = kwargs['dim']
        else:
            if self.flag_debug:
                print("Did not get the dimension of lpf, defaulting to 3.")
            self.dim = 3
        
        self.filterState_ = np.zeros(self.dim)
        self.filterStateDer_ = np.zeros(self.dim)
        return

    def proceed_state(self, input, dt):
        det = self.gainP_ * dt * dt + self.gainQ_ * dt + 1.
        stateDer = (self.filterStateDer_+self.gainP_*dt*input)/det - \
            (dt*self.gainP_*self.filterState_)/det
        self.filterState_ = (dt*(self.filterStateDer_+self.gainP_*dt*input))/det + \
            ((dt*self.gainQ_+1.)*self.filterState_)/det
        self.filterStateDer_ = stateDer
        return

    def reset_state(self):
        for ind in range(self.dim):
            self.filterState_[ind] = 0.
            self.filterStateDer_[ind] = 0.
        return

    def reset_state(self, filterState, filterStateDer):
        self.filterState_ = filterState
        self.filterStateDer_ = filterStateDer
        return

# Controller base
class UAV_pid():
    def __init__(self, *args, **kwargs):
        # PID Controller Vehicle Parameters
        if 'debug' in kwargs:
            self.flag_debug = kwargs['debug']
        else:
            self.flag_debug = False
        
        if 'gravity' in kwargs:
            self.gravity_ = kwargs['gravity']
        else:
            if self.flag_debug:
                print("Did not get the gravity from the params, defaulting to 9.81 m/s^2")
            self.gravity_ = 9.81
        
        if 'vehicleMass' in kwargs:
            self.vehicleMass_ = kwargs['vehicleMass']
        else:
            if self.flag_debug:
                print("Did not get the vehicle mass from the params, defaulting to 1.0 kg")
            self.vehicleMass_ = 1.0
        
        if 'vehicleInertia' in kwargs:
            self.vehicleInertia_ = kwargs['vehicleInertia']
        else:
            if self.flag_debug:
                print("Did not get the PID inertia from the params, defaulting to [0.0049, 0.0049. 0.0069] kg m^2")
            self.vehicleInertia_ = np.array([0.0049, 0.0049, 0.0069])
        
        if 'momentArm' in kwargs:
            self.momentArm_ = kwargs['momentArm']
        else:
            if self.flag_debug:
                print("Did not get the PID moment arm from the params, defaulting to 0.08 m")
            self.momentArm_ = 0.08

        if 'thrustCoeff' in kwargs:
            self.thrustCoeff_ = kwargs['thrustCoeff']
        else:
            if self.flag_debug:
                print("Did not get the PID thrust coefficient from the params, defaulting to 1.91e-6 N/(rad/s)^2")
            self.thrustCoeff_ = 1.91e-6

        if 'torqueCoeff' in kwargs:
            self.torqueCoeff_ = kwargs['torqueCoeff']
        else:
            if self.flag_debug:
                print("Did not get the PID torque coefficient from the params, defaulting to 2.6e-7 Nm/(rad/s)^2")
            self.torqueCoeff_ = 2.6e-7
        
        if 'motorRotorInertia' in kwargs:
            self.motorRotorInertia_ = kwargs['motorRotorInertia']
        else:
            if self.flag_debug:
                print("Did not get the PID torque coefficient from the params, defaulting to 2.6e-7 Nm/(rad/s)^2")
            self.motorRotorInertia_ = 6.62e-6
        
        if 'motorTimeConstant' in kwargs:
            self.motorTimeConstant_ = kwargs['motorTimeConstant']
        else:
            if self.flag_debug:
                print("Did not get the PID torque coefficient from the params, defaulting to 2.6e-7 Nm/(rad/s)^2")
            self.motorTimeConstant_ = 0.02
        
        return
     
    def thrust_mixing(self, angAccCommand, thrustCommand):
        # Compute torque and thrust vector
        momentThrust = np.array([
            self.vehicleInertia_[0]*angAccCommand[0],
            self.vehicleInertia_[1]*angAccCommand[1],
            self.vehicleInertia_[2]*angAccCommand[2],
            -thrustCommand])
        
        # # Compute signed, squared motor speed values
        # motorSpeedsSquared = np.array([
        #     momentThrust[0]/(4*self.momentArm_*self.thrustCoeff_) + (-momentThrust[1])/(4*self.momentArm_*self.thrustCoeff_) + \
        #         (-momentThrust[2])/(4*self.torqueCoeff_) + momentThrust[3]/(4*self.thrustCoeff_),
        #     momentThrust[0]/(4*self.momentArm_*self.thrustCoeff_) + momentThrust[1]/(4*self.momentArm_*self.thrustCoeff_) +  \
        #         momentThrust[2]/(4*self.torqueCoeff_) + momentThrust[3]/(4*self.thrustCoeff_),
        #     (-momentThrust[0])/(4*self.momentArm_*self.thrustCoeff_) + momentThrust[1]/(4*self.momentArm_*self.thrustCoeff_) + \
        #         (-momentThrust[2])/(4*self.torqueCoeff_)+ momentThrust[3]/(4*self.thrustCoeff_),
        #     (-momentThrust[0])/(4*self.momentArm_*self.thrustCoeff_) + (-momentThrust[1])/(4*self.momentArm_*self.thrustCoeff_) + \
        #         momentThrust[2]/(4*self.torqueCoeff_) + momentThrust[3]/(4*self.thrustCoeff_)
        # ])

        G1xy = self.thrustCoeff_ * self.momentArm_
        G1z = self.torqueCoeff_
        G1t = self.thrustCoeff_
        G2z = self.motorRotorInertia_ / self.motorTimeConstant_

        invG1xy = 1./(4.*G1xy)
        invG1z = 1./(4.*G1z)
        invG1t = 1./(4.*G1t)

        invG1 = np.zeros((4,4))
        invG1 = np.array([
            [ invG1xy,  invG1xy, -invG1z, -invG1t],
            [-invG1xy,  invG1xy,  invG1z, -invG1t],
            [-invG1xy, -invG1xy, -invG1z, -invG1t],
            [ invG1xy, -invG1xy,  invG1z, -invG1t]
        ])
    
        # Initial estimate of commanded motor speed using only G1
        motorSpeedsSquared = invG1.dot(momentThrust)

        # Compute signed motor speed values
        propSpeedCommand = np.zeros(4)
        for i in range(4):
            propSpeedCommand[i] = np.copysign(np.sqrt(np.fabs(motorSpeedsSquared[i])), motorSpeedsSquared[i])
        
        return propSpeedCommand

# PID angular rate controller
class UAV_pid_angular_rate(UAV_pid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # PID Controller Gains (roll / pitch / yaw)
        if 'propGain' in kwargs:
            self.propGain_ = kwargs['propGain']
        else:
            if self.flag_debug:
                print("Did not get the PID gain p from the params, defaulting to 9.0")
            self.propGain_ = np.array([9.0, 9.0, 9.0])

        if 'intGain' in kwargs:
            self.intGain_ = kwargs['intGain']
        else:
            if self.flag_debug:
                print("Did not get the PID gain i from the params, defaulting to 3.0")
            self.intGain_ = np.array([3.0, 3.0, 3.0])
        
        if 'derGain' in kwargs:
            self.derGain_ = kwargs['derGain']
        else:
            if self.flag_debug:
                print("Did not get the PID gain d from the params, defaulting to 0.3")
            self.derGain_ = np.array([0.3, 0.3, 0.3])

        # PID Controller Integrator State and Bound
        self.intState_ = np.array([0., 0., 0.])
        if 'intBound' in kwargs:
            self.intBound_ = kwargs['intBound']
        else:
            if self.flag_debug:
                print("Did not get the PID integrator bound from the params, defaulting to 1000.0")
            self.intBound_ = np.array([1000., 1000., 1000.])

        return
    
    def control_update(self, angVelCommand, thrustCommand, curval, curder, dt):
        angAccCommand = np.zeros(3)
        stateDev = angVelCommand - curval

        self.intState_ += dt*stateDev
        self.intState_ = np.fmin(np.fmax(-self.intBound_,self.intState_),self.intBound_)
        angAccCommand = self.propGain_*stateDev + \
            self.intGain_*self.intState_ - self.derGain_*curder
        propSpeedCommand = self.thrust_mixing(angAccCommand, thrustCommand)
        return propSpeedCommand
    
    def reset_state(self):
        self.intState_ = np.zeros(3)
        return

# PID position controller
class UAV_pid_waypoint(UAV_pid_angular_rate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # PID Controller Gains (x, y, z)
        if 'positionGain' in kwargs:
            self.position_gain = kwargs['positionGain']
        else:
            self.position_gain = np.array([7., 7., 7.])

        if 'velocityGain' in kwargs:
            self.velocity_gain = kwargs['velocityGain']
        else:
            self.velocity_gain = np.array([3., 3., 3.])
        
        if 'integratorGain' in kwargs:
            self.integrator_gain = kwargs['integratorGain']
        else:
            self.integrator_gain = np.array([0., 0., 0.])
        
        if 'attitudeGain' in kwargs:
            self.attitude_gain = kwargs['attitudeGain']
        else:
            self.attitude_gain = np.array([10., 10., 10.])
        
        if 'thrustDirection' in kwargs:
            self.thrust_dir = kwargs['thrustDirection']
        else:
            self.thrust_dir = np.array([0., 0., -1.])

        if 'maxAcceleration' in kwargs:
            self.max_acceleration = kwargs['maxAcceleration']
        else:
            self.max_acceleration = 3.0

        if 'maxAngrate' in kwargs:
            self.max_angrate = kwargs['maxAngrate']
        else:
            self.max_angrate = 8.0

        if 'maxSpeed' in kwargs:
            self.max_speed = kwargs['maxSpeed']
        else:
            self.max_speed = 3.0
        
        self.max_velocity_poserror = self.max_speed*(self.velocity_gain/self.position_gain)
        self.position_error_integrator = np.zeros(3)

        return
    
    def saturateVector(self, vec, bound):
        if isinstance(bound, np.ndarray):
            ret_vec = copy.deepcopy(vec)
            bound_t = np.squeeze(bound)
            for i in range(bound_t.shape[0]):
                ret_vec[i] = max(-bound_t[i], min(vec[i], bound_t[i]))
            return ret_vec
        else:
            return np.fmax(-bound, np.fmin(vec, bound))
    
    def get_control(self, pos_err, att_cur, curvel, att_ref):
        # getAccelerationCommand
        sat_pos_err = self.saturateVector(pos_err, self.max_velocity_poserror)
        acc_cmd = self.position_gain*sat_pos_err \
                  - self.velocity_gain*curvel \
                  + self.integrator_gain*self.position_error_integrator
        
        # saturateVector
        acc_cmd = self.saturateVector(acc_cmd, self.max_acceleration)
        acc_cmd[2] -= 9.81
        thrust_cmd = self.vehicleMass_*acc_cmd
        
        # getAttitudeCommand
        thrustcmd_yawframe = quat_rotate(att_ref, thrust_cmd)
        thrust_rot = vecvec2quat(self.thrust_dir, thrustcmd_yawframe)
        att_cmd = mul_quat(att_ref, thrust_rot)
        
        # getAngularRateCommand
        att_error = mul_quat(inv_quat(att_cur), att_cmd)
        if att_error[0] < 0.:
            att_error *= -1.
        angle_error = quat2Euler(att_error)
        angrate_cmd = angle_error*self.attitude_gain
        
        scalar_thrust = np.linalg.norm(thrust_cmd)
        angrate_cmd = self.saturateVector(angrate_cmd, self.max_angrate)
        res = dict()
        res["angularrate"] = angrate_cmd
        res["thrust"] = scalar_thrust
        return res
        
    def control_update(self, command, curpos, curvel, curatt, curattVel, curattAcc, dt):
        # Get position offsets
        del_x = command[0] - curpos[0]
        del_y = command[1] - curpos[1]
        del_z = command[2] - curpos[2]
        pos_err = np.array([del_x, del_y, del_z])
        self.position_error_integrator += dt*pos_err

        if command.size == 4:
            yaw_ref = command[3]
        else:
            # yaw_ref = quat2Euler(curatt)[2]
            yaw_ref = 0.0

        att_ref = Euler2quat(np.array([0,0,yaw_ref]))
        att_cur = Euler2quat(np.array([curatt[0],curatt[1],curatt[2]]))
        
        res = self.get_control(pos_err, att_cur, curvel, att_ref)
        attVelCommand = np.array([res["angularrate"][0],res["angularrate"][1],res["angularrate"][2]])

        stateDev = attVelCommand - curattVel
        self.intState_ += dt*stateDev
        self.intState_ = np.fmin(np.fmax(-self.intBound_,self.intState_),self.intBound_)
        angAccCommand = self.propGain_*stateDev + \
            self.intGain_*self.intState_ - self.derGain_*curattAcc

        propSpeedCommand = self.thrust_mixing(angAccCommand, res["thrust"])

        return propSpeedCommand
    
    def reset_state(self):
        self.intState_ = np.zeros(3)
        self.position_error_integrator = np.zeros(3)
        return

# PID position controller
class UAV_pid_tracking(UAV_pid_angular_rate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Position PID Controller Gains (x, y, z)
        if 'positionPGain' in kwargs:
            self.position_p_gain = kwargs['positionPGain']
        else:
            self.position_p_gain = np.array([7., 7., 7.])

        if 'positionDGain' in kwargs:
            self.position_d_gain = kwargs['positionDGain']
        else:
            self.position_d_gain = np.array([3., 3., 3.])
        
        if 'positionIGain' in kwargs:
            self.position_i_gain = kwargs['positionIGain']
        else:
            self.position_i_gain = np.array([0., 0., 0.])
            
        # Velocity PD Controller Gains (x, y, z)
        if 'velocityPGain' in kwargs:
            self.velocity_p_gain = kwargs['velocityPGain']
        else:
            self.velocity_p_gain = np.array([6.,6.,4.5])

        if 'velocityDGain' in kwargs:
            self.velocity_d_gain = kwargs['velocityDGain']
        else:
            self.velocity_d_gain = np.array([.5,.5,.3])
        
        # Saturation boundary
        if 'maxVelocityCommand' in kwargs:
            self.maxVelocityCommand = kwargs['maxVelocityCommand']
        else:
            self.maxVelocityCommand = 4.
        
        if 'maxAccelerationCommand' in kwargs:
            self.maxAccelerationCommand = kwargs['maxAccelerationCommand']
        else:
            self.maxAccelerationCommand = 4.
        
        if 'attitudeGain' in kwargs:
            self.attitude_gain = kwargs['attitudeGain']
        else:
            self.attitude_gain = np.array([10., 10., 10.])
        
        if 'thrustDirection' in kwargs:
            self.thrust_dir = kwargs['thrustDirection']
        else:
            self.thrust_dir = np.array([0., 0., -1.])

        if 'maxAcceleration' in kwargs:
            self.max_acceleration = kwargs['maxAcceleration']
        else:
            self.max_acceleration = 3.0

        if 'maxAngrate' in kwargs:
            self.max_angrate = kwargs['maxAngrate']
        else:
            self.max_angrate = 8.0

        if 'maxSpeed' in kwargs:
            self.max_speed = kwargs['maxSpeed']
        else:
            self.max_speed = 3.0
        
        self.max_velocity_poserror = self.max_speed*(self.position_d_gain/self.position_p_gain)
        self.position_error_integrator = np.zeros(3)

        return
    
    def saturateVector(self, vec, bound):
        if isinstance(bound, np.ndarray):
            ret_vec = copy.deepcopy(vec)
            bound_t = np.squeeze(bound)
            for i in range(bound_t.shape[0]):
                ret_vec[i] = max(-bound_t[i], min(vec[i], bound_t[i]))
            return ret_vec
        else:
            return np.fmax(-bound, np.fmin(vec, bound))

    def control_update(self, traj_ref, curpos, curvel, curacc, curatt, curattVel, curattAcc, dt):
        # Get position offsets
        del_x = traj_ref[0] - curpos[0]
        del_y = traj_ref[1] - curpos[1]
        del_z = traj_ref[2] - curpos[2]
        pos_err = np.array([del_x, del_y, del_z])
        self.position_error_integrator += dt*pos_err
        
        del_vx = traj_ref[3] - curvel[0]
        del_vy = traj_ref[4] - curvel[1]
        del_vz = traj_ref[5] - curvel[2]
        vel_err = np.array([del_vx, del_vy, del_vz])

        # getVelocityCommand    
        sat_pos_err = self.saturateVector(pos_err, self.max_velocity_poserror)
        vel_cmd = self.position_p_gain*sat_pos_err \
                  + self.position_d_gain*vel_err \
                  + self.position_i_gain*self.position_error_integrator
        
        sat_vel_cmd = self.saturateVector(vel_cmd, self.maxVelocityCommand)
        
        # getAccelerationCommand
        vel_err += sat_vel_cmd
        
        del_ax = traj_ref[6] - curacc[0]
        del_ay = traj_ref[7] - curacc[1]
        del_az = traj_ref[8] - curacc[2]
        acc_err = np.array([del_ax, del_ay, del_az])
        
        acc_cmd = self.velocity_p_gain*vel_err \
                  + self.velocity_d_gain*acc_err
        
        acc_cmd = self.saturateVector(acc_cmd, self.maxAccelerationCommand)
        acc_cmd[2] -= 9.81
        thrust_cmd = self.vehicleMass_*acc_cmd
        
        # getAttitudeCommand
        yaw_ref = traj_ref[15]
        att_ref = Euler2quat(np.array([0,0,yaw_ref]))
        att_cur = Euler2quat(np.array([curatt[0],curatt[1],curatt[2]]))
        
        thrustcmd_yawframe = quat_rotate(att_ref, thrust_cmd)
        thrust_rot = vecvec2quat(self.thrust_dir, thrustcmd_yawframe)
        att_cmd = mul_quat(att_ref, thrust_rot)
        
        # getAngularRateCommand
        att_error = mul_quat(inv_quat(att_cur), att_cmd)
        if att_error[0] < 0.:
            att_error *= -1.
        angle_error = quat2Euler(att_error)
        angrate_cmd = angle_error*self.attitude_gain
        
        thrustCommand = np.linalg.norm(thrust_cmd)
        attVelCommand = self.saturateVector(angrate_cmd, self.max_angrate)

        stateDev = attVelCommand - curattVel
        self.intState_ += dt*stateDev
        self.intState_ = np.fmin(np.fmax(-self.intBound_,self.intState_),self.intBound_)
        angAccCommand = self.propGain_*stateDev + \
            self.intGain_*self.intState_ - self.derGain_*curattAcc

        propSpeedCommand = self.thrust_mixing(angAccCommand, thrustCommand)

        return propSpeedCommand
    
    def reset_state(self):
        self.intState_ = np.zeros(3)
        self.position_error_integrator = np.zeros(3)
        return

if __name__ == "__main__":
    # execute only if run as a script
    print("test")
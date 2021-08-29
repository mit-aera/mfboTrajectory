#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import os, sys, time, copy, argparse
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy import interpolate

def unwrap(ang):
    return (ang+np.pi)%(2*np.pi)-np.pi

def quat_wx2xw(quat):
    res = np.zeros(4)
    res[:3] = quat[1:]
    res[3] = quat[0]
    return res

def quat_xw2wx(quat):
    res = np.zeros(4)
    res[1:] = quat[:3]
    res[0] = quat[3]
    return res

def mul_quat(quat1, quat0):
    w0, x0, y0, z0 = quat0
    w1, x1, y1, z1 = quat1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                    x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                    -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                    x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def inv_quat(q):
    return np.array([q[0],-q[1],-q[2],-q[3]])/(q[0]**2+q[1]**2+q[2]**2+q[3]**2)

def vecvec2quat(vec1, vec2):
    quat = np.zeros(4)
    quat[0] = np.sqrt((vec1.dot(vec1))*(vec2.dot(vec2))) + vec1.dot(vec2);

    if quat[0] < 1e-6:
        quat[0] = 0.
        if abs(vec1[0]) > abs(vec1[2]):
            quat[1] = -vec1[1]
            quat[2] = vec1[0]
            quat[3] = 0.
        else:
            quat[1] = 0.;
            quat[2] = -vec1[2];
            quat[3] = vec1[1];
    else:
        quat[1] = vec1[1]*vec2[2] - vec1[2]*vec2[1];
        quat[2] = vec1[2]*vec2[0] - vec1[0]*vec2[2];
        quat[3] = vec1[0]*vec2[1] - vec1[1]*vec2[0];
    quat_norm = np.sqrt(quat[0]**2+quat[1]**2+quat[2]**2+quat[3]**2)
    quat /= quat_norm
    return quat

def quat_rotate(quat, vec):
    invNormSq = 1./(quat[0]**2+quat[1]**2+quat[2]**2+quat[3]**2)
    out = np.zeros(3)
    out[0] = (vec[0]*quat[0]**2 - 2.*vec[2]*quat[0]*quat[2] + 2.*vec[1]*quat[0]*quat[3] + 
              vec[0]*quat[1]**2 + 2.*vec[1]*quat[1]*quat[2] + 2.*vec[2]*quat[1]*quat[3] - 
              vec[0]*quat[2]**2 - vec[0]*quat[3]**2)*invNormSq

    out[1] = (vec[1]*quat[0]**2 + 2.*vec[2]*quat[0]*quat[1] - 2.*vec[0]*quat[0]*quat[3] - 
              vec[1]*quat[1]**2 + 2.*vec[0]*quat[1]*quat[2] + vec[1]*quat[2]**2 + 
              2.*vec[2]*quat[2]*quat[3] - vec[1]*quat[3]**2)*invNormSq

    out[2] = (vec[2]*quat[0]**2 - 2.*vec[1]*quat[0]*quat[1] + 2.*vec[0]*quat[0]*quat[2] - 
              vec[2]*quat[1]**2 + 2.*vec[0]*quat[1]*quat[3] - vec[2]*quat[2] + 
              2.*vec[1]*quat[2]*quat[3] + vec[2]*quat[3]**2)*invNormSq
    return out


def quat2Euler(q):
    # w x y z <- All of the quaternion is in this order except for scipy
    roll = np.arctan2(2*(q[0]*q[1]+q[2]*q[3]), (1.-2.*(np.power(q[1],2)+np.power(q[2],2))))
    sinp = 2.*(q[0]*q[2]-q[3]*q[1])
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi/2, sinp)
    else:
        pitch = np.arcsin(sinp)
    yaw = np.arctan2(2.*(q[0]*q[3]+q[1]*q[2]), (1.-2.*(np.power(q[2],2)+np.power(q[3],2))))

    # roll, pitch, yaw = unwrap(R.from_quat(quat_wx2xw(q)).as_euler('xyz', degrees=False))

    return roll, pitch, yaw

def Euler2quat(att):
    cr = np.cos(att[0]/2)
    sr = np.sin(att[0]/2)
    cp = np.cos(att[1]/2)
    sp = np.sin(att[1]/2)
    cy = np.cos(att[2]/2)
    sy = np.sin(att[2]/2)
    
    q = np.zeros(4)
    q[0] = cy * cp * cr + sy * sp * sr
    q[1] = sr * cp * cy - cr * sp * sy
    q[2] = cr * sp * cy + sr * cp * sy
    q[3] = cr * cp * sy - sr * sp * cy
    
    return q

# body->unity
def ned2enu(pos, att):
    r_bias2 = R.from_euler('x', 180, degrees=True)
    r_att = R.from_quat(quat_wx2xw(att))
    # P = np.array([
    #     [0,-1,0],
    #     [-1,0,0],
    #     [0,0,-1]
    # ])
    P = np.array([
        [0,1,0],
        [1,0,0],
        [0,0,-1]
    ])
    r_P = R.from_matrix(P)
    pos_t = P.dot(pos)
    att_t = quat_xw2wx(R.as_quat(r_P*r_att*r_bias2))
    # att_t = quat_xw2wx(R.as_quat(r_P*r_att))
    # att_t = mul_quat(quat_xw2wx(R.as_quat(r_P)),att)
    # R_att = R.as_matrix(R.from_quat(quat_wx2xw(att)))
    # att_t = quat_xw2wx(R.as_quat(R.from_matrix(P.dot(R_att))))

    return pos_t, att_t
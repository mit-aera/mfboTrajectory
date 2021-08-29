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
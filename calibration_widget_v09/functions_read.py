#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 19:01:39 2020

@author: sasha
"""

#DATASETS to take
dsets = {}
dsets['xes_2d_p64'] = {  'i0': "I0", 
                         'pips': 'PIPS',
                         'exposure_time': 'exposureTime', 
                         'images': 'lambdaOne_images',
                         'wheel_motor': 'motor_sample_wheel'}

def load_hdf5(file, data_desc):
    
    pass
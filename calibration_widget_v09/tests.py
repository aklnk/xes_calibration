#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:23:53 2020

@author: sasha
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt

import function_set as fs
from time import time

f = h5py.File('./r0296/r0296.hdf5', "r")
images = f.get('lambdaOne_images')

t0=time()
rois = fs.find_rois(images[-1], roi_cnt=8, output=False)
peaks = fs.find_emission_positions(images[0], rois, max_peak_cnt=1, output=False)
fs.fit_emission_positions(images[0], rois, peaks)
t1=time()
print(t1-t0)
# print(rois)
# print(peaks)
plt.imshow(images[0], vmax=250)
plt.show()

for r in rois:
    plt.plot(np.sum(images[0,r[0]:r[1]], axis=0))
plt.show()
            
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 11:13:24 2020

@author: kalinko
"""
import os
import sys
import h5py
import numpy as np
from scipy.integrate import simps
from scipy.interpolate import Rbf, UnivariateSpline
#
from PyQt5 import QtWidgets as QtGui
from PyQt5 import QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D

from lmfit.models import PseudoVoigtModel

r = 5 #eV

run = 296
fit_file = -1
folder = './r{:04d}/'.format(run)
# folder = '/gpfs/processed/r{:04d}'.format(run)
files = os.listdir(folder)
files.sort()

# #remove all files except .xes
files = [f for f in files if f.endswith('.xes')]

x, y = np.genfromtxt(folder + files[fit_file], usecols=(0,1), unpack=True)

mod = PseudoVoigtModel()
pars = mod.guess(y, x=x)
out = mod.fit(y, pars, x=x)
# print(out.fit_report(min_correl=0.25))
params = out.params.valuesdict()
central_energy = params['center']
print(central_energy)

# out.plot()

plt.plot(x, y)
plt.plot(x, out.best_fit)
plt.show()

energy = []
herfd = []

for f in files:
    if f.endswith('.xes'):
        spl = f.split('_')
        energy.append(float(spl[-1][:-4]))
        
        x, y = np.genfromtxt(folder + f, usecols=(0,1), unpack=True)
        
        w = np.where(np.logical_and(x>central_energy-r, x<central_energy+r))
        
        spl = UnivariateSpline(x[w], y[w])
        val = spl(central_energy)
        herfd.append(val)
        # plt.plot(x[w], y[w])
        # plt.plot(central_energy, val)
        # plt.show()


print(energy, val)
plt.plot(energy, herfd, '.')
plt.show()
np.savetxt('r{:04d}_fit_herfd.txt'.format(run), np.transpose([energy, herfd]))


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:15:03 2020

@author: sasha
"""

import numpy as np
# import scipy as sp
import scipy.signal as sps
from lmfit.models import GaussianModel, PseudoVoigtModel
from lmfit import Parameters



def find_rois(image, roi_cnt=1, roi_r=5, sort=True, output=False):
    
    hsum = np.sum(image, axis=1)
    hpeaks, _ = sps.find_peaks(hsum)
    hprom, _, _ = sps.peak_prominences(hsum, hpeaks)
    hsort = np.argsort(hprom)
    hsort = np.flip(hsort)
    
    rois = []
    for i in range(roi_cnt):
        rois.append([hpeaks[hsort][i]-roi_r, hpeaks[hsort][i]+roi_r])
        if output: print('Suggested ROIs', rois[-1][0], rois[-1][1])
        
    if sort: rois.sort()
    
    return rois
    

def find_emission_positions(image, rois, max_peak_cnt=1, output=False):        
    #for each roi get peak positions
    positions = []
    for r in rois:    
        vsum = np.sum(image[r[0]:r[1]], axis=0)    
        vpeaks, _ = sps.find_peaks(vsum)
        vprom, _, _ = sps.peak_prominences(vsum, vpeaks)
        vsort = np.argsort(vprom)
        vsort = np.flip(vsort)
        peaks_roi=[]
        for i in range(max_peak_cnt):
            peaks_roi.append(vpeaks[vsort][i])
        if output: print('found peaks', peaks_roi)
        positions.append(peaks_roi)
    return positions

def fit_emission_positions(image, rois, positions):
    for r, p in zip(rois, positions):
        x = np.arange(image.shape[1])
        y = np.sum(image[r[0]:r[1]], axis=0)
        mod = GaussianModel()
        # print(mod.param_names)
        # pars = mod.guess(y, x=x)
        # 
        # pars['amplitude'].value = p[0]
        # print(pars)
        pars = Parameters()
        pars.add('amplitude', value=1)
        pars.add('center', value=p[0])
        pars.add('sigma', value=1)
        out = mod.fit(y, pars, x=x)
        # print(out.fit_report())
        # print(pdict)
        print(out.params['center'].value)
        # for name, param in out.params.items():
        #     print('{:7s} {:11.5f} {:11.5f}'.format(name, param.value, param.stderr))
    pass
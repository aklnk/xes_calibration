# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:32:08 2018

@author: angerpos
"""

import numpy as np
import transform as trf
import h5py
from time import time


class Image:
    '''
    Docstring comes here
    '''
    
    ENERGY = "mono_energy"
    ENERGY1 = "monoPos"
    I0 = "i0"
    I0_1 = 'I0'
    NAME_OLD = "lambda_images"     # OLD VERSION FILES
    NAME_OLD1 = "lambda_image"
    NAME_NEW = "lambdaOne_images" # NEW VERSION FILES
    NAME_GREATEYE = "data"
    NAME_PINK_GREATEYE = "/RAW/GE_Raw_Image"
    NAME_JNGFR_ON = "jngfr_data_on"
    NAME_JNGFR_OFF = "jngfr_data_off"
    NAME_JNGFR_TR = "jngfr_data_transient"
    
    def __init__(self, path, imnr):
        
        self.path = path
        self.imnr = imnr
        if self.path != "":
            file = h5py.File(self.path, "r")
            #extract energy and i0 from file
            try:
                self.energy = np.array(file.get(Image.ENERGY), dtype=np.float64)[imnr]
            except:
                self.energy = 0
            try:
                self.energy = np.array(file.get(Image.ENERGY1), dtype=np.float64)[imnr]
            except:
                self.energy = 0
            try:
                self.i_0 = np.array(file.get(Image.I0), dtype=np.float64)[imnr]
            except:
                self.i_0 = 1
            try:
                self.i_0 = np.array(file.get(Image.I0_1), dtype=np.float64)[imnr]
            except:
                self.i_0 = 1
                
            file.close()
        else:
            self.energy = 0
            self.i_0 = 1
            
        #initialize empty image
        self.main = None
        
        self.flags = {'flat': False, 'black': False, 'i_0': False, 
                      'bad_px': False, 'normalized': False}
        
        return
    
    def update_properties(self):
        
        self.max_noise = trf.max_noise(self.main)
        self.std_noise = trf.std_noise(self.main)
    
    def load_image(self, reset=False, laser="on"):
        
        if self.main is not None and not reset:
            print("Image already loaded - pass")
            return
        start = time()
        file = h5py.File(self.path, "r")
        end = time()
        print("open hdf5 file time = ", end-start)
        
        start = time()
        hdf5_keys = file.keys()
        
#        #Determine old or new file naming format and load image in memory
#        if any(key == Image.NAME_OLD for key in hdf5_keys):
#            self.ver = 0
#            self.main = np.array(file.get(Image.NAME_OLD))[self.imnr]
#        elif any(key == Image.NAME_NEW for key in hdf5_keys):
#            self.ver = 1
#            self.main = np.array(file.get(Image.NAME_NEW), dtype=np.float64)[self.imnr]
#        elif any(key ==Image.NAME_GREATEYE for key in hdf5_keys):
#            self.ver = 2
#            self.main = np.array(file.get(Image.NAME_GREATEYE), dtype=np.float64)
#        elif any(key == "RAW" for key in hdf5_keys):
#            self.ver = 3
#            self.main = np.array(file.get(Image.NAME_PINK_GREATEYE), dtype=np.float64)[self.imnr]
#            print(self.main.shape)
#        else:
#            for key in hdf5_keys:
#                if len(file.get(key)) == 3:
#                    self.main = np.array(file.get(key), dtype=np.float64)[self.imnr]
#                    break
#                
#            print("Error: no valid image found in file")
        
        #Determine old or new file naming format and load image in memory
        if Image.NAME_OLD in hdf5_keys:
            self.ver = 0
            dset = file.get(Image.NAME_OLD)
            self.main = np.array(dset[self.imnr], dtype=np.float64)
#            self.main = np.fliplr(self.main)
        elif Image.NAME_OLD1 in hdf5_keys:
            self.ver = 0
            dset = file.get(Image.NAME_OLD1)
            self.main = np.array(dset[self.imnr], dtype=np.float64)
#            self.main = np.fliplr(self.main)
        elif Image.NAME_NEW in hdf5_keys:
            self.ver = 1
            dset = file.get(Image.NAME_NEW)
            self.main = np.array(dset[self.imnr], dtype=np.float64)
        elif Image.NAME_GREATEYE in hdf5_keys:
            self.ver = 2
            dset = file.get(Image.NAME_GREATEYE)
            self.main = np.array(dset[self.imnr], dtype=np.float64)
        elif "RAW" in hdf5_keys:
            self.ver = 3
            dset = file.get(Image.NAME_PINK_GREATEYE)
            self.main = np.array(dset[self.imnr], dtype=np.float64)
            print(self.main.shape)
        elif Image.NAME_JNGFR_ON in hdf5_keys or Image.NAME_JNGFR_OFF in hdf5_keys:
            self.ver = 4
            if laser=="on":
                dset = file.get(Image.NAME_JNGFR_ON)
            if laser=="off":
                dset = file.get(Image.NAME_JNGFR_OFF)
            if laser=="tr":
                dset = file.get(Image.NAME_JNGFR_TR)
            self.main = np.array(dset[self.imnr], dtype=np.float64)
            
        else:
            for key in hdf5_keys:
                if len(file.get(key)) == 3:
                    self.main = np.array(file.get(key), dtype=np.float64)[self.imnr]
                    break
                
            print("Error: no valid image found in file")
        
        for key in self.flags.keys():
            self.flags[key] = False
        
        file.close()
        
        end = time()
        print("open image time = ", end-start)
            
        
        start = time()
        if np.sum(self.main)!=0:
            self.update_properties()
        end = time()
        print("update properties time = ", end-start)
    
    
    def rescale(self, factor):
        '''Divide image by a constant factor (e.g. I_0 correction).'''
        self.main = self.main / factor
        self.update_properties()
        
    def gradient_correct(self, **kwargs):
        
        mask = trf.gradient_mask(self.main, **kwargs)
        
        for index in np.argwhere(mask):
            trf.average_neighbors(self.main, mask, index)
            
    def black_correct(self, mask):
        
        for index in np.argwhere(mask):
            trf.average_neighbors(self.main, mask, index)        
    
    def find_roi(self, num, peak_kwargs={}, region_kwargs={}):
        '''
        Find ROIs from image, combining previous methods.
        !!! List possible kwargs !!!
        Output: region indices (start, end) (list of lists)
        '''
        
        b_w_y = trf.project_image(trf.cut_noise(self.main, self.max_noise), axis=1)
#        for debugging
#        import matplotlib.pyplot as plt
#        plt.figure(dpi=300)
#        plt.plot(b_w_y, linewidth=0.1)
#        plt.show()   
        peaks, sig = trf.best_peaks(b_w_y, num, **peak_kwargs)
        
        print(peaks, sig)
        
        return trf.find_regions(b_w_y, peaks, **region_kwargs)
    
    def find_angles(self, region_inds, **kwargs):
        
        #print(region_inds)
        angles = []
        del_xs = []
        for reg in region_inds:
            #print("#############################")
            #print(reg)
            
            try: angle, del_x = trf.find_angle(self.main, reg, **kwargs)
            except:         #TODO Which exception?
                angle = 0.0
            angles.append(angle)
            del_xs.append(del_x)
            #print(angle)
            #print(del_x)
            
        return angles, del_xs
    
    def find_positions(self, region_inds, angles, refine=True, reg_kwargs={}, peak_kwargs={}):
            
        positions = []
        for reg, angle in zip(region_inds, angles):
            pos = trf.find_elastic_position(self.main, reg, angle, refine=refine, reg_kwargs=reg_kwargs, peak_kwargs=peak_kwargs)
            positions.append(pos)      
        
        return positions
    
    def refine_roi(self):
        pass

    def cut_roi(self, region_inds, angles, del_xs, refine_inds=None):
        
        if not refine_inds:
            refine_inds = np.zeros_like(region_inds, dtype=int)
        
        self.roi = []
        for reg, angle, del_x, ref in zip(region_inds, angles, del_xs, refine_inds):
            roi = trf.rotate_region(self.main, reg, angle, del_x)
            self.roi.append(roi[ref[0]:roi.shape[0]-ref[1]])
        return
    
    def display_in_console(self):
        import matplotlib.pyplot as plt
        
        if self.main is not None:
            plt.figure(dpi=200)
            plt.imshow(self.main, cmap='gist_ncar')
            plt.show()
        return
    
    
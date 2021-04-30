# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 13:03:28 2017

@authors: sasha & angerpos
"""

import sys
import os
import signal
import warnings
from sys import exit, argv, version

import numpy as np
import collections as col

from scipy.interpolate import InterpolatedUnivariateSpline, Rbf

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import __version__ as mpl_version

import h5py

from PIL import Image

from init import QTVer
import data

import zmq

from time import sleep, time

# from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool as ThreadPool
from functools import partial

def receive_signal(signum, stack):
    print('Received:', signum)
    
signal.signal(2, receive_signal)


__path__=[os.path.dirname(os.path.abspath(__file__))]
warnings.simplefilter('ignore', np.RankWarning)


def show_exception_and_exit(exc_type, exc_value, tb):
    import traceback
    traceback.print_exception(exc_type, exc_value, tb)
#    raw_input("Press key to exit.")
#    sys.exit(-1)

sys.excepthook = show_exception_and_exit

LASER = ""

#Image dimensions
# LAMBDA 516 x 1556
# imVert = 516
# imHor = 1556

#PILATUS 100K 195 x 487
imVert = 195
imHor = 487

#PILATUS 300K 618 x 487
# imVert = 618
# imHor = 487

#GREAT EYE 128 x 512
#imVert = 128
#imHor = 512

#PINK GREAT EYE
#imVert = 255
#imHor = 1024

#JungFrau
#imVert = 250
#imHor = 1024
#LASER = "off"


# Correction flags - will be applied upon opening an image
BLACK_IMAGE_CORRECION = False
FLAT_IMAGE_CORRECION = False
I0_CORRECTION = True
AUTO_GRADIENT_CORRECTION = True
NORMALIZE = True
REBIN = False
binFactor = 3

if REBIN:
    imHor = int(imHor/binFactor)

XANES_SMOOTH = False #not supportet at the moment



LOAD_FROM_LAMBDA = False
LOAD_FROM_PILATUS_100K = False

if LOAD_FROM_LAMBDA: #define tango devices
    from PyTango import DeviceProxy, DevState
    lmbdOne = DeviceProxy("p64/lambda750k/01")
    
    port = "5489"
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind("tcp://*:%s" % port)
    
if LOAD_FROM_PILATUS_100K: #define tango devices
    
    port = "5489"
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind("tcp://*:%s" % port)



#file names for corrections
BLACK_IMAGE = "lmdb-one-8500-et4500-dark.tif"
FLAT_IMAGE = "lmdb-one-8500-et4500-ff.tif"

if QTVer == 4:
    from PyQt4 import QtGui, QtCore
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
    print("Python version: " + str(version))
    print("Qt version: " + str(QtCore.qVersion()))
    print("MatPlotLib version: " + str(mpl_version))
    print("h5py version: " + h5py.version.version)
    print("HDF5 version: " + h5py.version.hdf5_version)
    
if QTVer == 5:
    from PyQt5 import QtWidgets as QtGui
    from PyQt5 import QtCore
    from PyQt5 import QtGui as QtG
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    print("Python version: " + str(version))
    print("Qt version: " + str(QtCore.qVersion()))
    print("MatPlotLib version: " + str(mpl_version))
    print("h5py version: " + h5py.version.version)
    print("HDF5 version: " + h5py.version.hdf5_version)
    
def gaus(x,a,x0,sigma, bg):
    return (a*np.exp(-(x-x0)**2/(2*sigma**2))) + bg
    
def rebin_data(energy, fluo, Emin, Emax, step):
    
    if(energy[-1] < energy[0]):
#        energy = np.fliplr([energy])[0]
#        fluo = np.fliplr([fluo])[0]
        spl = InterpolatedUnivariateSpline(energy[::-1], fluo[::-1])
    else:
        spl = InterpolatedUnivariateSpline(energy, fluo)
    
    new_energy_scale = np.arange(Emin, Emax, step)
    
    y_rbf = spl(new_energy_scale)

    return new_energy_scale, y_rbf
    
def rebin_exafs(k, exafs, kmin, kmax, dk):
    spl = InterpolatedUnivariateSpline(k, exafs)
    newk = np.arange(kmin, kmax, dk)
    y_rbf = spl(newk)
    return newk, y_rbf

def calibrate_worker(key, params):
    print(os.getpid())
    print(key)
    start = time()
    forAveraging = []
    forSave = []
    commonE = []  

    #depending on the crystal and orientation used calculate d spacing
    refl = params['reflection']
    crystal = params['crystal']
    calibrate_parms = params['calibrate_parms']
    calibration_positions = params['calibration_positions']
    calibration_energies = params['calibration_energies']
    calibration_angles = params['calibration_angles']
    energy_scale = params['energy_scale']
    images = params['images']
    image = images[key]
    header = "" # string with used extraction and calibration parameters (roi, pixel-energy positions)
    header = "Energy="+'{:.2f}\n'.format(image.energy)
    columns = np.arange(0, image.main.shape[1], 1)
    if len(refl)>3:
        refl = refl.split(',')
        print(refl)
    else:
        refl = list(refl)
    if(crystal =='Si'):
        dspace = 5.4307 / np.sqrt(int(refl[0])**2 + int(refl[1])**2 + int(refl[2])**2)
#            print("D space = ", dspace)
    if(crystal =='Ge'):
        dspace = 5.658 / np.sqrt(int(refl[0])**2 + int(refl[1])**2 + int(refl[2])**2)
    if(crystal =='SiO2'): #only for 10-12
        dspace = 4.564 / 2
#            a=4.91304
#            c=5.40463
#            dspace = 5.4307 / np.sqrt(int(refl[0])**2 + int(refl[1])**2 + int(refl[2])**2)
    
    header = header + "Crystal=" + crystal + "\n"
    header = header + "Reflection=" + params['reflection'] + "\n"
    
    calibrationCoeficients = []
    
    enscales = []
    vertSums = []
    
    
    for i in range(len(calibration_positions[0])):
        posit = []
        ang = []
        header = header + "ROI="+str(i)+"="+   str(calibrate_parms['region_inds'][i][0])+"=" + \
                                                str(calibrate_parms['region_inds'][i][1]) + "=" + \
                                                str(calibrate_parms['angles'][i]) + "=" + \
                                                str(calibrate_parms['del_x'][i]) + "\n"
                                                
                                               
        for j in range(len(calibration_energies)):
            if calibration_positions[j] != []:
                posit.append(calibration_positions[j][i])
                ang.append(calibration_angles[j])
                header = header + "    Pos="+str(i)+"="+str(j)+"="+ str(calibration_energies[j])+"="+ \
                                                                str(calibration_positions[j][i]) + "\n"
                                                                
        
        
        a, b = np.polyfit(posit, ang, 1)
        calibrationCoeficients.append([a,b])
        
        angle_scale = columns*calibrationCoeficients[-1][0]+ calibrationCoeficients[-1][1]
        # print("Angle scale ", angle_scale[0], angle_scale[-1])

        enscale = 1239.84187 / ((2*dspace/10) * np.sin(np.radians(angle_scale)))
#            print("Energy scale ", enscale[0], enscale[-1])

        roi = image.main[int(calibrate_parms['region_inds'][i][0]):int(calibrate_parms['region_inds'][i][1])]
        
        vertSum = np.sum(roi, axis=0) #/ av_nr
        
#            print(roi.shape, self.calibrate_parms['region_inds'][i][0],self.calibrate_parms['region_inds'][i][1])#, enscale[0],enscale[-1])
        
        if(enscale[-1]<enscale[0]):
            enscales.append(enscale[::-1])
        else:
            enscales.append(enscale)
        vertSums.append(vertSum)
    
    
    #Determine E_min, E_max automatically
    if energy_scale['min'] == '':
        emin = np.max([enscale[0] for enscale in enscales])
    else: 
        emin = float(energy_scale['min'])
    if energy_scale['max'] == '':
        emax = np.min([enscale[-1] for enscale in enscales])
    else:
        emax = float(energy_scale['max'])
    if energy_scale['de'] == '':
        de = (emax - emin) / imHor
    else:
        de = float(energy_scale['de'])
    
    for enscale, vertSum in zip(enscales, vertSums):
        #print(enscale)
        commonE, rdata = rebin_data(enscale, vertSum, emin, emax, de)
        forAveraging.append(rdata)
        
        
    average =  np.sum(forAveraging, axis=0) / len(forAveraging)
       
    
    # TODO test os independence
    savefile = os.path.join(os.path.split(image.path)[0],
                            key + ".xes") # + "_" + str(self.current_img.imnr).zfill(4) + "_" + "{:.2f}".format(self.current_img.energy) + '.xes'
    #print(savefile)
    forSave.append(commonE)
    forSave.append(average)
    forSave = np.concatenate((forSave, forAveraging))
    np.savetxt(savefile, np.transpose(forSave), header = header)
    
    end = time()
    print("calibration time =", end-start)
    return commonE, average

###############################################################################
    
class listener(QtCore.QObject):   
    message = QtCore.pyqtSignal(str)    
    
    def __init__(self, parent=None):
        QtCore.QObject.__init__(self, parent)
        print("Starting listener ...")
        
    def listen(self):
        while True:
            print("Listener started ...")
            msg = socket.recv_string()
            self.message.emit(msg)
        pass
    
###############################################################################


class calibrationWidget(QtGui.QWidget):
    
    
    def __init__(self, parent=None):
        super(calibrationWidget, self).__init__()

        self.calibrationEnergies = []
        self.calibrationPositions = []
        self.calibrationWaveLength = []
        self.calibrationAngles = []
        # ROI display
        self.rectangles = []
        # (odered) list of keys and corresponding image objects in dict
        # TODO: self.keys not necessary when using OrderedDict
        self.keys = []
        self.images = {}

        self.zmax = 1.0
        
        self.iNum = 0
        
        self.initUI()
        
        self.measurementName = "currentMeasurement"
        
        
        self.restoreSettings()
        # OrderedDict of calibration parameters
        self.calibrate_parms = col.OrderedDict()
        self.calibrate_parms['region_inds'] = []
        self.calibrate_parms['angles'] = []
        self.calibrate_parms['del_x'] = []
        #initialize empty variables
        self.flatData = None
        self.blackData = None
        self.rescaleFactor = None
        
        self.blackData = np.array([np.array(Image.open(BLACK_IMAGE))])[0]        
        self.blackMask = np.zeros_like(self.blackData, dtype=np.bool)
    
        for index in np.argwhere(self.blackData > 0.0):
            self.blackMask[tuple(index)] = True

        self.flatData = np.array([np.array(Image.open(FLAT_IMAGE))])[0]  
        print(self.flatData)
        print(self.flatData.shape)
        
        if LOAD_FROM_LAMBDA or LOAD_FROM_PILATUS_100K: #start thread to read images from Lambda live view
            self.listenerThread = listener()

            
            self.wthread = QtCore.QThread(self)        
            self.listenerThread.moveToThread(self.wthread)
            self.wthread.started.connect(self.listenerThread.listen)
#            self.rdffnd.finished.connect(self.RDFinished)
            self.listenerThread.message.connect(self.on_listener_message)
#            self.rdffnd.finished.connect(self.wthread.quit)
#            print("Starting optimisation...")
            self.wthread.start()
#            print("optimisation started...")
            
    def on_listener_message(self, message):
        #process message
        energy = -1
        i0 = 1
        split_message = message.split("=")
        for m in split_message:
            if(m[0]=='N'): #change measurement name
                self.measurementName = m[1:]
                return
            if(m[0]=='E'): #Energy
                energy = m[1:]
            if(m[0]=='I'): #I0 intensity
                i0 = m[1:] 
            if(m[0]=='P'): #Pilatus full file Path
                pilatusImagePath = m[1:]
                

        
        #add new measurement to list
        if LOAD_FROM_LAMBDA:
            self.readLambda(energy, i0)
        if LOAD_FROM_PILATUS_100K:
            self.readPilatus(energy, i0, pilatusImagePath)
            
        
    def initUI(self):
        
        #Main layout to add all other widgets and layouts
        loutMain = QtGui.QGridLayout()
        
        #File operations layout
        self.btnOpenCalibrationFiles = QtGui.QPushButton("Open file(s) ... ")
        self.btnOpenCalibrationFiles.clicked.connect(self.openData)
        
        self.btnRemoveCalibrationFiles = QtGui.QPushButton("Remove file ... ")
        self.btnRemoveCalibrationFiles.setToolTip("WARNING: Might cause errors.")
        self.btnRemoveCalibrationFiles.clicked.connect(self.removeData)              
        
        lblCalibrationFiles = QtGui.QLabel("Calibration Files")
        self.lstCalibrationFiles = QtGui.QListWidget()
        self.lstCalibrationFiles.itemClicked.connect(self.lstCalibrationFilesItemClicked)
        self.lstCalibrationFiles.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        
        loutFiles = QtGui.QGridLayout()
        
        loutFiles.addWidget(lblCalibrationFiles,                 0, 0)
        loutFiles.addWidget(self.btnOpenCalibrationFiles,        1, 0)
        loutFiles.addWidget(self.btnRemoveCalibrationFiles,      1, 1)
        loutFiles.addWidget(self.lstCalibrationFiles,            2, 0, 1, 2)
        
        #Crystal Layout
        loutCrystal = QtGui.QGridLayout()
        
        self.cmbCrystal = QtGui.QComboBox()
        self.cmbCrystal.addItems(['Si', 'Ge', 'SiO2'])
        lblCrystalReflection = QtGui.QLabel("Crystal Reflection used")
        self.edtCrystalReflection = QtGui.QLineEdit("440")
        self.edtCrystalReflection.returnPressed.connect(self.recalcAngles)
        
        self.edtSearchArea = QtGui.QLineEdit('')
        self.edtSearchArea.setPlaceholderText("0, 516 (Specify Area)") 
        self.edtSearchArea.setToolTip("Default: (0,516)")
        
        self.edtHorSearchArea = QtGui.QLineEdit('')
        self.edtHorSearchArea.setPlaceholderText("0, 1556 (Specify Area)") 
        self.edtHorSearchArea.setToolTip("Default: (0,1556)")

        loutCrystal.addWidget(self.cmbCrystal,           0, 0)
        loutCrystal.addWidget(lblCrystalReflection,      0, 1)
        loutCrystal.addWidget(self.edtCrystalReflection, 0, 2)
        loutCrystal.addWidget(self.edtSearchArea,        1, 0, 1, 3)
        loutCrystal.addWidget(self.edtHorSearchArea,        2, 0, 1, 3)

        #Gradient Correction Layout
        self.btnGradientCorrection = QtGui.QPushButton("GradientCorrection")
        self.btnGradientCorrection.clicked.connect(self.gradientCorrection)   
        
        self.btnResetImage = QtGui.QPushButton("Reset Image")
        self.btnResetImage.clicked.connect(self.resetImage)
        
        self.edtGradientThreshold = QtGui.QLineEdit("")
        self.edtGradientThreshold.setPlaceholderText("5.0") 
        self.edtGradientThreshold.setToolTip("Threshold scale for triggering bad pixels. Too low values might lead to false-positives!")
 
        loutGradCorrect = QtGui.QGridLayout()         
        loutGradCorrect.addWidget(self.btnGradientCorrection,           0, 0)
        loutGradCorrect.addWidget(self.edtGradientThreshold,            0, 1)
        loutGradCorrect.addWidget(self.btnResetImage,                   0, 2)
        
        #Complete left hand side layout
        loutOperations = QtGui.QGridLayout()
        
        loutOperations.addLayout(loutFiles,                 0, 0)
        loutOperations.addLayout(loutCrystal,               2, 0)
        loutOperations.addLayout(loutGradCorrect,           3, 0)

        
        
        #Calibration Widget Layout (calibration parameters and energies/positions)
        #
        #Find regions layout
        self.btnfindRegions = QtGui.QPushButton("Find Regions")
        self.btnfindRegions.clicked.connect(self.findRegions)
        self.btnfindRegions.setToolTip("Find specified number of ROIs")
        
        self.btnFindPositions = QtGui.QPushButton("Find Positions")
        self.btnFindPositions.clicked.connect(self.findPositions)
        self.btnFindPositions.setToolTip("Find elastic line positions for all ROIs")
        
        self.edtFindRegionsNum = QtGui.QLineEdit()
        self.edtFindRegionsNum.setPlaceholderText("4 (No. of regions)") 
        self.edtFindRegionsNum.setToolTip("Desired number of ROIs")
        
        self.edtFindRegionsThreshold = QtGui.QLineEdit()
        self.edtFindRegionsThreshold.setPlaceholderText("20.0 (Threshold)")
        self.edtFindRegionsThreshold.setToolTip("Control width of ROIs - higher value means narrower ROIs (no large effect at the moment)")
        
        self.btnFindAngles = QtGui.QPushButton("Find Angles")
        self.btnFindAngles.clicked.connect(self.findAngles) 
        self.btnFindAngles.setToolTip("Optimize orientation of each ROI - may take a few seconds")
        
#        self.btnOpenRegions = QtGui.QPushButton("**Open Regions")
#        self.btnOpenRegions.setToolTip("Currently not working...")
        #self.btnOpenRegions.clicked.connect(self.openRegions)
        
#        self.btnSaveRegions = QtGui.QPushButton("**Save Regions")
#        self.btnSaveRegions.setToolTip("Currently not working...")
        #self.btnSaveRegions.clicked.connect(self.saveRegions)
        
        self.btnLoadCalibration = QtGui.QPushButton("Load Calibration")
        self.btnLoadCalibration.setToolTip("Currently not working...")
        self.btnLoadCalibration.clicked.connect(self.loadCalibration)
        
        loutFindReg = QtGui.QGridLayout()
        
        loutFindReg.addWidget(self.btnfindRegions,              0, 0)
        loutFindReg.addWidget(self.btnFindPositions,            0, 1)
        loutFindReg.addWidget(self.edtFindRegionsNum,           1, 0)
        loutFindReg.addWidget(self.edtFindRegionsThreshold,     1, 1)
        loutFindReg.addWidget(self.btnFindAngles,               2, 0, 1, 2)
        loutFindReg.addWidget(self.btnLoadCalibration,          3, 0, 1, 2)
#        loutFindReg.addWidget(self.btnSaveRegions,              3, 1)        

        #Display/edit regions layout        
        self.cmbRegions = QtGui.QComboBox()
        self.cmbRegions.activated.connect(self.plotROI)
        
        self.lstRegions = QtGui.QListWidget()
        self.lstRegions.itemClicked.connect(self.lstRegionsItemClicked)
        self.lstRegions.itemDoubleClicked.connect(self.lstRegionsItemDoubleClicked)
        self.lstRegions.itemChanged.connect(self.lstRegionsItemChanged)
        
        self.btnAddRegion = QtGui.QPushButton("+")
        self.btnAddRegion.clicked.connect(self.addRegion)
        self.btnRemoveRegion = QtGui.QPushButton("-")
        self.btnRemoveRegion.clicked.connect(self.removeRegion)
       
        self.lstAngles = QtGui.QListWidget()
        self.lstAngles.itemClicked.connect(self.lstAnglesItemClicked)
        self.lstAngles.itemDoubleClicked.connect(self.lstAnglesItemDoubleClicked)
        self.lstAngles.itemChanged.connect(self.lstAnglesItemChanged)

        loutDispReg = QtGui.QGridLayout()
        loutDispReg.addWidget(QtGui.QLabel("Regions"),               0, 0)
        loutDispReg.addWidget(QtGui.QLabel("Angles"),                0, 3)
        loutDispReg.addWidget(self.btnAddRegion,                     0, 1)
        loutDispReg.addWidget(self.btnRemoveRegion,                  0, 2)   
        loutDispReg.addWidget(self.lstRegions,                       1, 0, 1, 3)
        loutDispReg.addWidget(self.lstAngles,                        1, 3, 1, 3)
        
        lblEnergies = QtGui.QLabel("Energies")
        self.lstEnergies = QtGui.QListWidget()
        self.lstEnergies.itemClicked.connect(self.lstEnergiesItemClicked)
        self.lstEnergies.itemDoubleClicked.connect(self.lstEnergiesItemDoubleClicked)
        self.lstEnergies.itemChanged.connect(self.lstEnergiesItemChanged)
        
        lblPositions = QtGui.QLabel("Positions")
        self.lstPositions = QtGui.QListWidget()
        self.lstPositions.itemDoubleClicked.connect(self.lstPositionsItemDoubleClicked)
        self.lstPositions.itemChanged.connect(self.lstPositionsItemChanged)
        self.lstPositions.itemClicked.connect(self.lstPositionsItemClicked)

        self.btnAddPosition = QtGui.QPushButton("+")
        self.btnAddPosition.clicked.connect(self.addPosition)
        self.btnRemovePosition = QtGui.QPushButton("-")
        self.btnRemovePosition.clicked.connect(self.removePosition)
       
        self.btnAddEnergy = QtGui.QPushButton("+")
        self.btnAddEnergy.clicked.connect(self.addEnergy)
        self.btnRemoveEnergy = QtGui.QPushButton("-")
        self.btnRemoveEnergy.clicked.connect(self.removeEnergy)
        
        loutDispEnergy = QtGui.QGridLayout()
        loutDispEnergy.addWidget(lblEnergies,                           0, 0)
        loutDispEnergy.addWidget(lblPositions,                          0, 3)
        loutDispEnergy.addWidget(self.btnAddEnergy,                     0, 1)
        loutDispEnergy.addWidget(self.btnRemoveEnergy,                  0, 2) 
        loutDispEnergy.addWidget(self.btnAddPosition,                   0, 4)
        loutDispEnergy.addWidget(self.btnRemovePosition,                0, 5) 
        loutDispEnergy.addWidget(self.lstEnergies,                      1, 0, 1, 3)
        loutDispEnergy.addWidget(self.lstPositions,                     1, 3, 1, 3)

#        self.btnSaveCalibration = QtGui.QPushButton("**Save Calibration")
#        self.btnSaveCalibration.setToolTip("Currently not working...")
#        self.btnSaveCalibration.clicked.connect(self.saveCalibration)


        #Energy range and step size
        lblEmin = QtGui.QLabel("Emin")
        lblEmax = QtGui.QLabel("Emax")
        lbldE = QtGui.QLabel("dE")
        
        self.edtEmin = QtGui.QLineEdit("")
        self.edtEmin.setPlaceholderText("Auto")
        self.edtEmin.setToolTip("Lower bound of calibrated energy range")
        self.edtEmax = QtGui.QLineEdit("")
        self.edtEmax.setPlaceholderText("Auto")
        self.edtEmax.setToolTip("Upper bound of calibrated energy range")
        self.edtdE = QtGui.QLineEdit("0.02")
        self.edtdE.setPlaceholderText("Auto")
        self.edtdE.setToolTip("Energy step")
        
        self.btnCalibrate = QtGui.QPushButton("Calibrate")
        self.btnCalibrate.setToolTip("Calibrate selected image and save as .xes file")
        self.btnCalibrate.clicked.connect(self.calibrate)
        
        self.btnCalibrateAll = QtGui.QPushButton("CalibrateAll")
        self.btnCalibrateAll.setToolTip("Calibrate all loaded images and save as .xes files")
        # self.btnCalibrateAll.clicked.connect(self.calibrateAll)  
        self.btnCalibrateAll.clicked.connect(self.calibrateAll_parallel)
        
        self.btnCalibrateSelected = QtGui.QPushButton("Calibrate selected")
        self.btnCalibrateSelected.setToolTip("Calibrate selectedimages and save as .xes files")
        self.btnCalibrateSelected.clicked.connect(self.calibrateSelected)
        
        loutCalPrm = QtGui.QGridLayout()
        loutCalPrm.addWidget(lblEmin,                   0, 0)
        loutCalPrm.addWidget(lblEmax,                   0, 1)
        loutCalPrm.addWidget(lbldE,                     0, 2)
        
        loutCalPrm.addWidget(self.edtEmin ,             1, 0)
        loutCalPrm.addWidget(self.edtEmax ,             1, 1)
        loutCalPrm.addWidget(self.edtdE ,               1, 2)

        loutCalPrm.addWidget(self.btnCalibrate,         2, 0, 1, 3)
        loutCalPrm.addWidget(self.btnCalibrateAll,      3, 0, 1, 3)
        loutCalPrm.addWidget(self.btnCalibrateSelected, 4, 0, 1, 3)
#        loutCalPrm.addWidget(self.btnSaveCalibration,   3, 0, 1, 3)
#        loutCalPrm.addWidget(self.btnLoadCalibration,   4, 0, 1, 3)
        
        loutCalPrm.setColumnStretch(0, 2)
        loutCalPrm.setColumnStretch(1, 2)
        loutCalPrm.setColumnStretch(2, 3)

        loutCalibration = QtGui.QGridLayout()  
        
        loutCalibration.addWidget(QtGui.QLabel("Calibration"),   0, 0)
        loutCalibration.addLayout(loutFindReg,                   1, 0)
        loutCalibration.addLayout(loutDispReg,                   1, 1)      
        loutCalibration.addLayout(loutDispEnergy,                1, 2)
        loutCalibration.addLayout(loutCalPrm,                    1, 3)

        
        tight_layout0 = {'top': 0.99,
                        'bottom': 0.035,
                        'left': 0.035,
                        'right': 1.0,
                        'hspace': 0.2,
                        'wspace': 0.2}
        
        # Image display tab
        tabImage = QtGui.QWidget()
        
        # a figure instance to plot on
        self.fig = plt.figure(0)
        self.ax_image = plt.subplot() 
        self.fig.subplots_adjust(**tight_layout0)
        self.canv = FigureCanvas(self.fig)
        
        self.tbar = NavigationToolbar(self.canv, self)        
        loutFig = QtGui.QGridLayout()
        loutFig.addWidget(self.tbar,                            0, 0, 1, 2)
        # TODO add image information (max noise, etc.) at the top of image tab for easier troubleshooting
#        placeholder_message = "Info about the image (e.g. noise lvl, scaling) \
#        will be displayed here."
#        loutFig.addWidget(QtGui.QLabel(placeholder_message),    0, 1)
        loutFig.addWidget(self.canv,                            1, 0, 1, 2)
        
        loutFig.setColumnStretch(0, 1)
        loutFig.setColumnStretch(1, 1)

        for item in ([self.ax_image.title, self.ax_image.xaxis.label, self.ax_image.yaxis.label] +
             self.ax_image.get_xticklabels() + self.ax_image.get_yticklabels()):
            item.set_fontsize(20)
        
        self.fig.canvas.mpl_connect('button_press_event', self.imageDoubleClick)
        
        self.vert_lines = []
        
        tabImage.setLayout(loutFig)
        
        #Region of interest display tab
        tabROIs = QtGui.QWidget()
        
        tight_layout1 = {'top': 0.984,
                'bottom': 0.032,
                'left': 0.019,
                'right': 0.991,
                'hspace': 0.281,
                'wspace': 0.014}
        
        self.fig1 = plt.figure(1)
        self.ax_roi = plt.subplot(6, 2, (1,2))
        self.ax_project = plt.subplot(6, 2, (3,12))
        self.fig1.subplots_adjust(**tight_layout1)
        self.canv1 = FigureCanvas(self.fig1)
        self.tbar1 = NavigationToolbar(self.canv1, self)      
        loutFig1 = QtGui.QVBoxLayout()
        loutFig1.addWidget(self.tbar1)
        loutFig1.addWidget(self.canv1)
        
        tabROIs.setLayout(loutFig1)
        
        # Calibration display tab
        tabCalibration = QtGui.QWidget()
        
        self.fig2 = plt.figure(2)
        self.ax_calibratedImage = plt.subplot()        
        self.canv2 = FigureCanvas(self.fig2)
        self.tbar2 = NavigationToolbar(self.canv2, self)    
        
        loutFig2 = QtGui.QGridLayout()
        
        self.btnClearCalibrationCanvas = QtGui.QPushButton("Clear Canvas")
        self.btnClearCalibrationCanvas.clicked.connect(self.clearCalibrationCanvas)
        
        loutFig2.addWidget(self.tbar2,                        0, 0)
        loutFig2.addWidget(self.btnClearCalibrationCanvas,    0, 1)
        loutFig2.addWidget(self.canv2,                        1, 0, 1, 2)
        
        loutFig2.setColumnStretch(0, 5)
        loutFig2.setColumnStretch(1, 1)        
        
        tabCalibration.setLayout(loutFig2)
        
        #XANES display tab
        #TODO currently without function
        tabXanes = QtGui.QWidget()        
        
        self.fig3 = plt.figure(3)
        self.ax_xanes = plt.subplot()        
        self.canv3 = FigureCanvas(self.fig3)
        self.tbar3 = NavigationToolbar(self.canv3, self)   
        
        self.edtregionXanes = QtGui.QLineEdit()
        self.btnCalcXanes = QtGui.QPushButton("Calculate XANES")   
        self.btnCalcXanes.clicked.connect(self.calculateXanes)
        
        loutFig3 = QtGui.QGridLayout()
        
        loutFig3.addWidget(self.tbar3,                                      0, 0, 1, 3)
        loutFig3.addWidget(self.canv3,                                      1, 0, 1, 3)
        loutFig3.addWidget(QtGui.QLabel("Fluo energy region for XANES :"),  2, 0)

        loutFig3.addWidget(self.edtregionXanes,                             2, 1)
        loutFig3.addWidget(self.btnCalcXanes,                               2, 2)

        tabXanes.setLayout(loutFig3)
        
        #Add everything together
        tabs = QtGui.QTabWidget()
        
        tabs.addTab(tabImage, "Image")
        tabs.addTab(tabROIs, "ROIs")
        tabs.addTab(tabCalibration, "Calibration")
        tabs.addTab(tabXanes, "XANES")
        #tabs.addTab(tabAdvSettings, "Adv. Settings")
        
        loutMain.addLayout(loutOperations,                  0, 0, 2, 1)
        loutMain.addWidget(tabs,                            0, 1)        
        loutMain.addLayout(loutCalibration,                 1, 1)
        
        loutMain.setColumnStretch(0, 1)
        loutMain.setColumnStretch(1, 6)
        loutMain.setRowStretch(0, 20)
        loutMain.setRowStretch(1, 1)
      
        self.setLayout(loutMain)
        #display default image
        self.image = self.ax_image.imshow(np.zeros((imVert, imHor)), vmin = 0, vmax = self.zmax, aspect='auto', interpolation='none',cmap='gist_ncar')
       
    def correctImage(self, current_img):
        
        if BLACK_IMAGE_CORRECION and not current_img.flags['black']:
            current_img.black_correct(self.blackMask)
            current_img.flags['black'] = True
            current_img.update_properties()
            
        if AUTO_GRADIENT_CORRECTION and not current_img.flags['bad_px']:
            threshold = float(self.edtGradientThreshold.placeholderText())

            try: threshold = float(self.edtGradientThreshold.text())
            except: print("Could not retrieve gradient threshold. Use default value {}".format(threshold))
              
            grad_parm = {'scale':threshold}
            current_img.gradient_correct(**grad_parm)
            current_img.flags['bad_px'] = True
            current_img.update_properties()
            
        if FLAT_IMAGE_CORRECION and not current_img.flags['flat']:   
            print(current_img.main.shape)
            print(current_img.main)
            current_img.main = current_img.main / self.flatData
            print(current_img.main)
            current_img.flags['flat'] = True
            current_img.update_properties()
            
        if I0_CORRECTION and not current_img.flags['i_0']:
            current_img.rescale(current_img.i_0)
            current_img.flags['i_0'] = True
        
        if NORMALIZE and not current_img.flags['normalized']:
            if self.rescaleFactor is None:
                self.rescaleFactor = np.max(current_img.main)
                
            current_img.rescale(self.rescaleFactor)
            current_img.flags['normalized'] = True
           
    
    def lstCalibrationFilesItemClicked(self):
        
        #retrieve Image object and load image
        cnt = self.lstCalibrationFiles.currentRow()
        key = self.keys[cnt]
        
        self.current_img = self.images[key]
        
        start = time()
        self.current_img.load_image(laser=LASER)
        end = time()
        print("opening time = ", end-start)
        
        self.correctImage(self.current_img)
        
        if REBIN and self.current_img.main.shape[1]>int(1556/binFactor):
#            height = int(self.main.shape[0])
#            width = int(self.main.shape[1])
#            vnew = int(self.main.shape[0]/3)
#            hnew = int(self.main.shape[1]/3)
#            N=3
            mainShape = self.current_img.main.shape[0]
            binShape = self.current_img.main.shape[1]
            
            while (binShape / binFactor != int(binShape/binFactor)):
                self.current_img.main = np.delete(self.current_img.main,1,axis=1)
                binShape = self.current_img.main.shape[1]
            
            self.current_img.main = np.reshape(self.current_img.main, (mainShape,int(binShape/binFactor),binFactor))
#            self.current_img.main = np.reshape(self.current_img.main, (516,389,4))
#            self.main = np.ravel(self.main)
#            print(self.main.shape)
            self.current_img.main = np.sum(self.current_img.main, axis=-1)
#            print(self.main.shape)
#            new_h = np.array_split(self.main, width//N, axis=1)
#            new_h_av = np.average(new_h, axis=1)
#            print(new_h_av.shape)
        
#            self.main = np.average(
#                    np.array_split(
#                    np.average(
#                    np.array_split(self.main, width // N, axis=1), axis=-1), height // N, axis=1), axis=-1)
           
#            self.main = np.array_split(self.main,width // N,axis=1)
#            print(self.main)
#            print(len(self.main))
#            print(new[0], new[0].shape, np.average(new[0], axis=-1), np.average(new[0], axis=-1).shape)
        
        
        self.image.set_data(self.current_img.main)

        # refresh canvas
        self.canv.draw()   
        
    def load_image(self, key):
        
        current_img = self.images[key]
        
        start = time()
        current_img.load_image(laser=LASER)
        end = time()
        print("opening time = ", end-start)
        
        start = time()
        self.correctImage(current_img)
        end = time()
        print("opening time = ", end-start)
        
        if REBIN and current_img.main.shape[1]>int(1556/binFactor):

            mainShape = current_img.main.shape[0]
            binShape = current_img.main.shape[1]
            
            while (binShape / binFactor != int(binShape/binFactor)):
                current_img.main = np.delete(current_img.main,1,axis=1)
                binShape = current_img.main.shape[1]
            
            current_img.main = np.reshape(current_img.main, (mainShape,int(binShape/binFactor),binFactor))

            current_img.main = np.sum(current_img.main, axis=-1)
    
        
    def readLambda(self, energy, i0):
#        filename
        energy=str(energy)
        i0=str(i0)
        key = "_".join([self.measurementName, str(self.iNum).zfill(4), '{:.2f}'.format(float(energy))])
        self.images[key] = data.Image("", 0)
        self.keys.append(key)
        
        self.lstCalibrationFiles.addItems(
                    [">".join([self.measurementName, str(self.iNum).zfill(4), '{:.2f}'.format(float(energy))])])
        
        self.iNum = self.iNum + 1
        
        #retrieve Image object and load image
        self.lstCalibrationFiles.setCurrentRow(len(self.lstCalibrationFiles))
        cnt = self.lstCalibrationFiles.currentRow()
        key = self.keys[cnt]

        self.current_img = self.images[key]
        self.current_img.energy = float(energy)
        self.current_img.i_0 = float(i0)
        
        ####### READ from Lambda here ################
        
#        self.current_img.main = np.zeros((imVert, imHor)) + self.iNum
        
        self.current_img.main = lmbdOne.LiveLastImageData
        
        ##################################
        
        self.current_img.update_properties()
        
        for key in self.current_img.flags.keys():
            self.current_img.flags[key] = False
            
    def readPilatus(self, energy, i0, imagePath):
#        filename
        energy=str(energy)
        i0=str(i0)
        key = "_".join([self.measurementName, str(self.iNum).zfill(4), '{:.2f}'.format(float(energy))])
        self.images[key] = data.Image("", 0)
        self.keys.append(key)
        
        self.lstCalibrationFiles.addItems(
                    [">".join([self.measurementName, str(self.iNum).zfill(4), '{:.2f}'.format(float(energy))])])
        
        self.iNum = self.iNum + 1
        
        #retrieve Image object and load image
        self.lstCalibrationFiles.setCurrentRow(len(self.lstCalibrationFiles))
        cnt = self.lstCalibrationFiles.currentRow()
        key = self.keys[cnt]

        self.current_img = self.images[key]
        self.current_img.energy = float(energy)
        self.current_img.i_0 = float(i0)
        
        ####### READ tif from Pilatus here ################
        
#        self.current_img.main = np.zeros((imVert, imHor)) + self.iNum
        
        self.current_img.main = Image.open(imagePath)
        
        ##################################
        
        self.current_img.update_properties()
        
        for key in self.current_img.flags.keys():
            self.current_img.flags[key] = False

    
    def openHDF5(self, path):
    
        file = h5py.File(path, 'r')
        HDF5_ENERGY = "mono_energy"
        HDF5_ENERGY1 = "monoPos"
        for key in file.keys():
            print(key)
        
        if any(key == data.Image.NAME_OLD for key in file.keys()):
            images = data.Image.NAME_OLD
        elif any(key == data.Image.NAME_OLD1 for key in file.keys()):
            images = data.Image.NAME_OLD1
        elif any(key ==data.Image.NAME_NEW for key in file.keys()):
            images = data.Image.NAME_NEW
        elif any(key == data.Image.NAME_GREATEYE for key in file.keys()):
            images = data.Image.NAME_GREATEYE
        elif any(key == data.Image.NAME_JNGFR_ON for key in file.keys()):
            if(LASER=="on"):
                images = data.Image.NAME_JNGFR_ON
            if(LASER=="off"):
                images = data.Image.NAME_JNGFR_OFF
            if(LASER=="tr"):
                images = data.Image.NAME_JNGFR_TR
        elif any(key == "RAW" for key in file.keys()):
            images = data.Image.NAME_PINK_GREATEYE
            
        number_of_images = 0
        if file.get(images).ndim == 2:
            number_of_images = 1
        if file.get(images).ndim == 3:
            number_of_images = len(file.get(images))
            array_of_images = np.array(file.get(images))

        
        for imnr in range(number_of_images):
            try:    energy = file.get(HDF5_ENERGY)[imnr]
            except: 
                try:    energy = file.get(HDF5_ENERGY1)[imnr]
                except: energy = 0
            key = "_".join([os.path.split(path)[1], str(imnr).zfill(4), '{:.2f}{}'.format(energy, LASER)])
            self.images[key] = data.Image(path, imnr)
            self.images[key].main = array_of_images[imnr]
            self.correctImage(self.images[key])
            self.keys.append(key)
            self.lstCalibrationFiles.addItems(
                    [">".join([os.path.split(path)[1], str(imnr).zfill(4), '{:.2f}{}'.format(energy, LASER)])])
            
        # for i in range( len(self.lstCalibrationFiles) ):
        #     start = time()
        #     print(os.linesep, "############################")
        #     print("loading image {0}/{1} ... ".format(i+1, len(self.lstCalibrationFiles)))    
        #     self.lstCalibrationFiles.setCurrentRow(i)
        #     self.lstCalibrationFilesItemClicked()
        #     end = time()
        #     print("total time =", end-start)
        
    def openData(self):
        
        filenames = []
        dlg = QtGui.QFileDialog()
        dlg.setFileMode(QtGui.QFileDialog.ExistingFiles)
        dlg.setAcceptMode(0) # open dialog
        # TODO currently only hdf5 files supported
        dlg.setNameFilters(["HDF5 files (*.hdf5)", "Text files (*.txt)", "DAT files (*.dat)", 
                            "TIFF images (*.tiff)", "All files (*.*)"])

        if dlg.exec_():
            filenames = dlg.selectedFiles()
        
        if filenames == []:
            return

        for fname in filenames:
            #print(fname)
            #print(os.path.splitext(fname))
            if os.path.splitext(fname)[1] == ".hdf5" or os.path.splitext(fname)[1] == ".h5":
                self.openHDF5(fname)
        
    def removeData(self):
        selected_indexes = self.lstCalibrationFiles.selectedIndexes()
        
        if len(selected_indexes) == 0:
            return
        
        selectedRows = [x.row() for x in selected_indexes]
        selectedRows.sort(reverse=True)
        
        print(selectedRows)
        
        for row in selectedRows:
            key = self.keys[row]
            del self.keys[row]
            del self.images[key]
            self.lstCalibrationFiles.takeItem(row)
        # TODO not working properly! 
#        cnt = self.lstCalibrationFiles.currentRow()
#        if cnt == -1: 
#            return
#        key = self.keys[cnt]
#
#        del self.keys[cnt]
#        del self.images[key]
#        self.lstCalibrationFiles.takeItem(cnt)
    
    def openRegions(self):
        # TODO not tested!
        self.cmbRegions.activated.disconnect(self.plotROI)
        
        filename = []
        dlg = QtGui.QFileDialog()
        dlg.setFileMode(QtGui.QFileDialog.ExistingFiles)
        dlg.setAcceptMode(0) # open dialog
        dlg.setNameFilters(["Region files (*.rgn)", "All files (*.*)"])
#        dlg.setDirectory(self.currentdir)
#        filenames = QStringList()
        if dlg.exec_():
            filename = dlg.selectedFiles()
        
        if filename == []:
            return
        self.calibrate_parms['region_inds'] = np.genfromtxt(filename[0], unpack=False)
        if self.calibrate_parms['region_inds'].ndim == 1:
            self.calibrate_parms['region_inds'] = np.array([self.calibrate_parms['region_inds']]) 
        self.rectangles = []
        for r in self.calibrate_parms['region_inds']: 
            if self.cmbRegions.count()%2==0:
                color = 'r'
            else:
                color = 'g'
                
            self.cmbRegions.addItem(str(r))
            
            
            rect = patches.Rectangle((r[0],r[2]),r[1]-r[0],r[3]-r[2],linewidth=1,edgecolor=color,facecolor='none')
            self.rectangles.append(rect)
            # Add the patch to the Axes
            self.ax_image.add_patch(rect)
            
            
        self.canv.draw()            
        self.cmbRegions.activated.connect(self.plotROI)
        
    def saveRegions(self):
        # TODO not tested!
        filename = []
        dlg = QtGui.QFileDialog()
        dlg.setFileMode(QtGui.QFileDialog.AnyFile)
        dlg.setAcceptMode(1) # save dialog
        dlg.setNameFilters(["Region files (*.rgn)", "All files (*.*)"])
        if dlg.exec_():
            filename = dlg.selectedFiles()
        
        if filename == []:
            return

        np.savetxt(filename[0], self.calibrate_parms['region_inds'])
              
    def gradientCorrection(self):
        #Get parameters
        threshold = float(self.edtGradientThreshold.placeholderText())

        try: threshold = float(self.edtGradientThreshold.text())
        except: print("Could not retrieve gradient threshold. Use default value {}".format(threshold))
                    
        grad_parm = {'scale':threshold}
        
        self.current_img.gradient_correct(**grad_parm)

        self.image.set_data(self.current_img.main)
        self.canv.draw()        
    
    def resetImage(self):
        #reset scaling factor to previous image and apply corrections anew
        self.rescaleFactor = None
        self.current_img.load_image(reset=True)
        self.correctImage()
        self.image.set_data(self.current_img.main)
        self.canv.draw() 
        
    def findRegions(self):     
        #Collect parameters
        #default
        num = 4
        region_kwargs = {}
        bounds = None
        peak_kwargs = {}
        
        if self.edtSearchArea.text() != '':
            try: 
                string = self.edtSearchArea.text().strip()
                bounds = [int(string.split(',')[0]), int(string.split(',')[1])]
            except:
                print("Could not retrieve specified area. Look everywhere for ROIs.")
        
        peak_kwargs['bounds'] = bounds
        
        if self.edtFindRegionsNum.text() != '':
            try: num = int(self.edtFindRegionsNum.text())
            except ValueError:
                print("Number of ROIs invalid. Use default value {}".format(num))

        if self.edtFindRegionsThreshold.text() != '':
            try: region_kwargs['thresh_scale'] = float(self.edtFindRegionsThreshold.text())
            except ValueError:
                print("ROI threshold invalid. Use default value {}".format(20.0))

        self.cmbRegions.clear()
        self.lstRegions.clear()
        
        self.calibrate_parms['region_inds'] = self.current_img.find_roi(num, region_kwargs=region_kwargs, peak_kwargs=peak_kwargs)

        print(self.calibrate_parms['region_inds'])

        for reg in self.calibrate_parms['region_inds']:
            self.cmbRegions.addItem(str(reg))
            self.lstRegions.addItem("{0}, {1}".format(reg[0], reg[1]))

        self.plotRegionBorders()
        #Add default angle 0.0 to all regions to enable plotting/calibration
        self.calibrate_parms['angles'] = [0.0] * len(self.calibrate_parms['region_inds'])
        self.calibrate_parms['del_x'] = [0] * len(self.calibrate_parms['region_inds'])
        
        self.lstAngles.clear()
        for angle in self.calibrate_parms['angles']:
            self.lstAngles.addItem("{}".format(angle))
        
    def plotRegionBorders(self):
        #remove old regions from image
        if self.rectangles != []: 
            for r in self.rectangles:
                r.remove()
        
        self.rectangles = []
        regions = self.calibrate_parms['region_inds']
        #regions = self.current_img.find_roi(num)
        regions = sorted(regions, key=lambda x: x[0])
        
        k=0        
        for reg in regions:
            # Create a Rectangle patch
            if k%2==0:
                color = 'r'
            else:
                color = 'g'
            k+=1
            rect = patches.Rectangle((0,reg[0]),self.current_img.main.shape[1]-0,reg[1]-reg[0],linewidth=1,edgecolor=color,facecolor='none')
            self.rectangles.append(rect)
            # Add the patch to the Axes
            self.ax_image.add_patch(rect)

        self.canv.draw()
        
    def findAngles(self):
        
        if not self.current_img.flags['bad_px']:
            self.gradientCorrection()
            self.canv.draw()    
        self.calibrate_parms['angles'], self.calibrate_parms['del_x'] = self.current_img.find_angles(self.calibrate_parms['region_inds'])
        
        self.lstAngles.clear()
        for angle in self.calibrate_parms['angles']:
            self.lstAngles.addItem("{}".format(angle))
       
    def findPositions(self):
            
        bounds = None
        if self.edtHorSearchArea.text() != '':
            try: 
                string = self.edtHorSearchArea.text().strip()
                bounds = [int(string.split(',')[0]), int(string.split(',')[1])]
            except:
                print("Could not retrieve specified area. Look everywhere for ROIs.")
        
#        peak_kwargs['bounds'] = bounds
        
        #default thresh for region search
        reg_kwargs = {'thresh_scale':10.0}
        peak_kwargs = {'bounds' : bounds}
        
        positions = self.current_img.find_positions(self.calibrate_parms['region_inds'], self.calibrate_parms['angles'], reg_kwargs=reg_kwargs, peak_kwargs=peak_kwargs)
#        print("Area Nr = ", i)    
#        print( np.transpose([peak_ind, sig]) )
#        print(peak_ind,len(peak_ind),type(peak_ind))

                
        #depending on the crystal and orientation used calculate d spacing
        refl = self.edtCrystalReflection.text()
        if len(refl)>3:
            refl = refl.split(',')
            print(refl)
        else:
            refl = list(refl)
        if(self.cmbCrystal.currentText() == 'Si'):
            dspace = 5.4307 / np.sqrt(int(refl[0])**2 + int(refl[1])**2 + int(refl[2])**2)
            print("D space = ", dspace)
        if(self.cmbCrystal.currentText() == 'Ge'):
            dspace = 5.658 / np.sqrt(int(refl[0])**2 + int(refl[1])**2 + int(refl[2])**2)

        #energy info available from hdf5 file
        try:      self.calibrationEnergies.append(self.current_img.energy)
        except:   self.calibrationEnergies.append(0)
        
        try:      self.calibrationWaveLength.append(1239.84187 / self.current_img.energy)
        except:   self.calibrationWaveLength.append(0)
        
        try:      self.calibrationAngles.append(np.degrees(np.arcsin(self.calibrationWaveLength[-1] / (2*dspace/10))))
        except:   self.calibrationAngles.append(0)

        self.calibrationPositions.append(positions)
        self.lstEnergies.addItem(str(self.calibrationEnergies[-1]))
        self.lstPositions.clear()
        self.lstPositions.addItems(["%.3f" % x for x in self.calibrationPositions[-1]])
        self.canv.draw()
        
        print(self.calibrationPositions)
        print(self.calibrationEnergies)


    def plotROI(self):        
        #cnt = self.cmbRegions.currentIndex()
        cnt = self.lstRegions.currentRow()

        self.current_img.cut_roi(self.calibrate_parms['region_inds'], 
                                 self.calibrate_parms['angles'], 
                                 self.calibrate_parms['del_x'])
        


        self.ax_roi.clear()
        self.ax_roi.imshow(self.current_img.roi[cnt], vmin=0, vmax=self.zmax, aspect='auto', interpolation='none', cmap='gist_ncar')
        self.ax_project.clear()
#        self.ax_project.plot(np.average(self.current_img.roi[cnt], axis=0), linewidth=0.75)
        self.ax_project.plot(np.sum(self.current_img.roi[cnt], axis=0), linewidth=0.75)
        self.ax_project.set_xlim([0, self.current_img.roi[cnt].shape[1]])
        self.canv1.draw()
        
        roi_save = np.sum(self.current_img.roi[cnt], axis=0)
        pixels = range(len(roi_save))
        np.savetxt("roi.dat", np.array([pixels,roi_save ]).T )


    def calibrateSelected(self):
        org_row = self.lstCalibrationFiles.currentRow()
        selected_indexes = self.lstCalibrationFiles.selectedIndexes()
        print(selected_indexes)
        for i in range(len(list(selected_indexes))):
            start = time()
            print(os.linesep, "############################")
            print("Calibrating image {0}/{1} ... ".format(i+1, len(list(selected_indexes)))) 
            y= selected_indexes[i].row()
            self.lstCalibrationFiles.setCurrentRow(y)
            self.lstCalibrationFilesItemClicked()
            self.calibrate(False)
            end = time()
            print("iteration time =", end-start)
            
        self.lstCalibrationFiles.setCurrentRow(org_row)
        self.lstCalibrationFilesItemClicked()
            

    def calibrateAll(self):
        start = time()
        org_row = self.lstCalibrationFiles.currentRow()
        
        for i in range( len(self.lstCalibrationFiles) ):
            
            print(os.linesep, "############################")
            print("Calibrating image {0}/{1} ... ".format(i+1, len(self.lstCalibrationFiles)))    
            self.lstCalibrationFiles.setCurrentRow(i)
            self.lstCalibrationFilesItemClicked()
            self.calibrate(False)
            
            
        self.lstCalibrationFiles.setCurrentRow(org_row)
        self.lstCalibrationFilesItemClicked()
        end = time()
        print("total time =", end-start)
        
    def calibrateAll_worker(self, i):
        print(i)
        # self.lstCalibrationFiles.setCurrentRow(i)
        # self.lstCalibrationFilesItemClicked()
        key = self.keys[i]
        self.load_image(key)   
        params = []
        params.append(self.images[key])
        params.append(i)
        self.calibrate(params)
        
    def calibrateAll_parallel(self):
        start = time()
        org_row = self.lstCalibrationFiles.currentRow()
        
        params = {}
        params['reflection'] = self.edtCrystalReflection.text()
        params['crystal'] = self.cmbCrystal.currentText()
        params['calibrate_parms'] = self.calibrate_parms
        params['calibration_positions'] = self.calibrationPositions
        params['calibration_energies'] = self.calibrationEnergies
        params['calibration_angles'] = self.calibrationAngles
        params['energy_scale'] = {'min':self.edtEmin.text(),
                                  'max': self.edtEmax.text(),
                                  'de': self.edtdE.text()}
        params['images'] = self.images
        
        print(len(self.calibrationPositions))
        
        with ThreadPool(processes=3) as pool:
            # pool.map(self.calibrateAll_worker, range(len(self.lstCalibrationFiles)))
            pool.map(partial(calibrate_worker, params=params), self.keys)
        
        # pool = ThreadPool(1)
        # pool.map(self.calibrateAll_worker, range(len(self.lstCalibrationFiles)))
        # pool.close()
        # pool.join()
        
        self.lstCalibrationFiles.setCurrentRow(org_row)
        self.lstCalibrationFilesItemClicked()
        end = time()
        print("calibration all time =", end-start)
        
        # org_row = self.lstCalibrationFiles.currentRow()
        
        # for i in range( len(self.lstCalibrationFiles) ):
        #     start = time()
        #     print(os.linesep, "############################")
        #     print("Calibrating image {0}/{1} ... ".format(i+1, len(self.lstCalibrationFiles)))    
        #     self.lstCalibrationFiles.setCurrentRow(i)
        #     self.lstCalibrationFilesItemClicked()
        #     self.calibrate()
        #     end = time()
        #     print("iteration time =", end-start)
            
        
        
    def calibrate(self, params):
        print(params)
        if params==False:
            current_img = self.current_img
            cnt = self.lstCalibrationFiles.currentRow()
        else:
            current_img = params[0]
            cnt = params[1]
        start = time()
        forAveraging = []
        forSave = []
        commonE = []
        columns = np.arange(0, current_img.main.shape[1], 1)
        
        header = "" # string with used extraction and calibration parameters (roi, pixel-energy positions)
        header = "Energy="+'{:.2f}\n'.format(current_img.energy)

        #depending on the crystal and orientation used calculate d spacing
        refl = self.edtCrystalReflection.text()
        if len(refl)>3:
            refl = refl.split(',')
            print(refl)
        else:
            refl = list(self.edtCrystalReflection.text())
        if(self.cmbCrystal.currentText() =='Si'):
            dspace = 5.4307 / np.sqrt(int(refl[0])**2 + int(refl[1])**2 + int(refl[2])**2)
#            print("D space = ", dspace)
        if(self.cmbCrystal.currentText() =='Ge'):
            dspace = 5.658 / np.sqrt(int(refl[0])**2 + int(refl[1])**2 + int(refl[2])**2)
        if(self.cmbCrystal.currentText() =='SiO2'): #only for 10-12
            dspace = 4.564 / 2
#            a=4.91304
#            c=5.40463
#            dspace = 5.4307 / np.sqrt(int(refl[0])**2 + int(refl[1])**2 + int(refl[2])**2)
        
        header = header + "Crystal=" + self.cmbCrystal.currentText() + "\n"
        header = header + "Reflection=" + self.edtCrystalReflection.text() + "\n"
        
        self.calibrationCoeficients = []
        
        enscales = []
        vertSums = []

        
        for i in range(len(self.calibrationPositions[0])):
            posit = []
            ang = []
            header = header + "ROI="+str(i)+"="+   str(self.calibrate_parms['region_inds'][i][0])+"=" + \
                                                    str(self.calibrate_parms['region_inds'][i][1]) + "=" + \
                                                    str(self.calibrate_parms['angles'][i]) + "=" + \
                                                    str(self.calibrate_parms['del_x'][i]) + "\n"
            for j in range(len(self.calibrationEnergies)):
                if self.calibrationPositions[j] != []:
                    posit.append(self.calibrationPositions[j][i])
                    ang.append(self.calibrationAngles[j])
                    header = header + "    Pos="+str(i)+"="+str(j)+"="+ str(self.calibrationEnergies[j])+"="+ \
                                                                    str(self.calibrationPositions[j][i]) + "\n"
            
            #print angles
#            print(ang)
#            print(posit)
#            for s in range(len(ang)-1):
#                d_det = (posit[s+1]-posit[s])/2
#                ang_rad = np.radians(ang)
#                R = d_det*0.055* np.tan(ang_rad[s]) * np.tan(ang_rad[s+1]) / ( np.tan(ang_rad[s+1])-np.tan(ang_rad[s]))
#                print("Calculated R", R)
#            d_det = (posit[-1]-posit[0])/2
#            ang_rad = np.radians(ang)
#            R = d_det*0.055* np.tan(ang_rad[0]) * np.tan(ang_rad[-1]) / ( np.tan(ang_rad[-1])-np.tan(ang_rad[0]))
#            print("Calculated R", R)
            
            a, b = np.polyfit(posit, ang, 1)
            self.calibrationCoeficients.append([a,b])
            
            angle_scale = columns*self.calibrationCoeficients[-1][0]+ self.calibrationCoeficients[-1][1]
            # print("Angle scale ", angle_scale[0], angle_scale[-1])

            enscale = 1239.84187 / ((2*dspace/10) * np.sin(np.radians(angle_scale)))
#            print("Energy scale ", enscale[0], enscale[-1])
    
            roi = current_img.main[int(self.calibrate_parms['region_inds'][i][0]):int(self.calibrate_parms['region_inds'][i][1])]
            
            vertSum = np.sum(roi, axis=0) #/ av_nr
            
#            print(roi.shape, self.calibrate_parms['region_inds'][i][0],self.calibrate_parms['region_inds'][i][1])#, enscale[0],enscale[-1])
            
            if(enscale[-1]<enscale[0]):
                enscales.append(enscale[::-1])
            else:
                enscales.append(enscale)
            vertSums.append(vertSum)
        
        
        #Determine E_min, E_max automatically
        if self.edtEmin.text() == '':
            emin = np.max([enscale[0] for enscale in enscales])
        else: 
            emin = float(self.edtEmin.text())
        if self.edtEmax.text() == '':
            emax = np.min([enscale[-1] for enscale in enscales])
        else:
            emax = float(self.edtEmax.text())
        if self.edtdE.text() == '':
            de = (emax - emin) / imHor
        else:
            de = float(self.edtdE.text())
        
        for enscale, vertSum in zip(enscales, vertSums):
            #print(enscale)
            commonE, rdata = rebin_data(enscale, vertSum, emin, emax, de)
            forAveraging.append(rdata)
            
            
        average =  np.sum(forAveraging, axis=0) / len(forAveraging)
#        print(len(average), len(commonE))
        self.ax_calibratedImage.plot(commonE, average)        
        
        # TODO test os independence
        savefile = os.path.join(os.path.split(current_img.path)[0],
                                self.keys[cnt] + ".xes") # + "_" + str(self.current_img.imnr).zfill(4) + "_" + "{:.2f}".format(self.current_img.energy) + '.xes'
        #print(savefile)
        forSave.append(commonE)
        forSave.append(average)
        forSave = np.concatenate((forSave, forAveraging))
        np.savetxt(savefile, np.transpose(forSave), header = header)

        self.canv2.draw()
        
        end = time()
        print("calibration time =", end-start)
    
    def clearCalibrationCanvas(self):
        self.ax_calibratedImage.clear()
        self.canv2.draw()
    
    def addPosition(self, value = 0):
        if len(self.calibrationEnergies) == 0:
            return
        else:
            energyNr = self.lstEnergies.currentRow()
            self.calibrationPositions[energyNr].append(value)
            self.lstEnergiesItemClicked()

    def removePosition(self):
        if len(self.calibrationEnergies) == 0:
            return
        else:
            energyNr = self.lstEnergies.currentRow()
            cntPos = self.lstPositions.currentRow()
            del self.calibrationPositions[energyNr][cntPos]
            self.lstEnergiesItemClicked()
        
    def addEnergy(self, value = 0):
        self.calibrationEnergies.append(float(value))
        self.calibrationWaveLength.append(float(value))
        self.calibrationAngles.append(float(value))
        
        self.calibrationPositions.append([])
        
        self.lstEnergies.addItem(str(self.calibrationEnergies[-1]))       
    
    def removeEnergy(self):
        energyNr = self.lstEnergies.currentRow()
        
        del self.calibrationEnergies[energyNr]
        del self.calibrationWaveLength[energyNr]
        del self.calibrationAngles[energyNr]        
        del self.calibrationPositions[energyNr]

        self.lstEnergies.clear()
        self.lstPositions.clear()
        
        self.lstEnergies.addItems((["%.3f" % x for x in self.calibrationEnergies]))
            
        self.lstEnergies.setCurrentRow(0)
        self.lstEnergiesItemClicked()
    
    def addRegion(self):
        
        string, ok = QtGui.QInputDialog.getText(self,
                                           "New ROI",
                                           "Region indices. Format: 'start, end'")
        if ok:
            try:
                string = string.strip()
                reglist = string.split(',')
                reg_start = int(reglist[0])
                reg_end = int(reglist[1])
            except:
                print("Could not retrieve new region indices. Please check input.")
                return
            
            self.calibrate_parms['region_inds'].append([reg_start, reg_end])
            self.calibrate_parms['region_inds'].sort(key=lambda x: x[0])
            self.lstRegions.clear()
            for reg in self.calibrate_parms['region_inds']:
                self.lstRegions.addItem("{0}, {1}".format(reg[0], reg[1]))

            self.plotRegionBorders()
            
        else:
            return
    
    def removeRegion(self):
        #TODO currently not working properly
        index = self.lstRegions.currentRow()
        print(index)
        print(self.calibrate_parms)
        
        for key in self.calibrate_parms.keys():
            print(self.calibrate_parms[key])
            del self.calibrate_parms[key][index]
            print(self.calibrate_parms[key])
        
        print(self.calibrate_parms)
        
        self.lstRegions.clear()
        for reg in self.calibrate_parms['region_inds']:
            self.lstRegions.addItem("{0}, {1}".format(reg[0], reg[1]))
        self.lstAngles.clear()
        for angle in self.calibrate_parms['angles']:
            self.lstAngles.addItem("{0}".format(angle))
            
        self.plotRegionBorders()
    
#    def saveCalibration(self):
#        #TODO not tested!
#        filename = []
#        dlg = QtGui.QFileDialog()
#        dlg.setFileMode(QtGui.QFileDialog.AnyFile)
#        dlg.setAcceptMode(1) # save dialog
#        dlg.setNameFilters(["Region files (*.clb)", "All files (*.*)"])
#        if dlg.exec_():
#            filename = dlg.selectedFiles()
#        
#        if filename == []:
#            return
#        print(self.calibrationPositions)    
#        save_array = []
#        save_array.append(self.calibrationEnergies)
#        save_array = save_array + list(np.transpose(np.asarray(self.calibrationPositions)))
#        
#        print(save_array)
#        
#
#        np.savetxt(filename[0], save_array)
        
        
    def loadCalibration(self):
        
        try:
            a = self.current_img.main.shape[1]
        except:
            msgBox = QtGui.QMessageBox()
            msgBox.setText('Open experimental file first.')
            msgBox.setIcon(QtGui.QMessageBox.Information)
            msgBox.exec_()
            return
        
        
        filename = []
        dlg = QtGui.QFileDialog()
        dlg.setFileMode(QtGui.QFileDialog.ExistingFiles)
        dlg.setAcceptMode(0) # open dialog
        dlg.setNameFilters(["XES containing calibration (*.xes)", "All files (*.*)"])

        if dlg.exec_():
            filename = dlg.selectedFiles()
        
        if filename == []:
            return
        
        roi = []
        cEnergies = []
        cPositions = []
        crystal = ""
        reflection = ""
        angles = []
        del_x = []
        
        with open(filename[0]) as f:
            for line in f:
                if line[0]=="#": #comment
                    if "Crystal" in line:
                        sp_line = line.split("=")
                        crystal = sp_line[-1].rstrip()
                    if "Reflection" in line:
                        sp_line = line.split("=")
                        reflection = sp_line[-1].rstrip()    
                    if "ROI" in line:
                        sp_line = line.split("=")
                        roi.append([int(sp_line[-4]), int(sp_line[-3])])
                        cEnergies.append([])
                        cPositions.append([])
                        angles.append(float(sp_line[-2]))
                        del_x.append(int(sp_line[-1]))
                    if "Pos" in line:
                        sp_line = line.split("=")
                        cEnergies[int(sp_line[1])].append(float(sp_line[-2]))
                        cPositions[int(sp_line[1])].append(float(sp_line[-1]))
                else:
                    break
                
        print(roi)
        print(cEnergies)
        print(np.transpose(cPositions))
    
        #set params        
        index = self.cmbCrystal.findText(crystal, QtCore.Qt.MatchFixedString)
        print(index)
        if index >= 0:
            self.cmbCrystal.setCurrentIndex(index)
        self.edtCrystalReflection.setText(reflection)
            
        self.calibrationEnergies = cEnergies[0]
        self.calibrationPositions = list(np.transpose(cPositions))
        
        self.calibrate_parms['region_inds'] = roi
        
        self.cmbRegions.clear()
        self.lstRegions.clear()
        for reg in self.calibrate_parms['region_inds']:
            self.cmbRegions.addItem(str(reg))
            self.lstRegions.addItem("{0}, {1}".format(reg[0], reg[1]))
            
            
        self.calibrate_parms['angles'] = angles
        self.calibrate_parms['del_x'] = del_x
        
        self.lstAngles.clear()
        for angle in self.calibrate_parms['angles']:
            self.lstAngles.addItem("{}".format(angle))
            
        self.plotRegionBorders()
        
        #depending on the crystal and orientation used calculate d spacing
        refl = self.edtCrystalReflection.text()
        if len(refl)>3:
            refl = refl.split(',')
        else:
            refl = list(refl)
        if(self.cmbCrystal.currentText() =='Si'):
            dspace = 5.4307 / np.sqrt(int(refl[0])**2 + int(refl[1])**2 + int(refl[2])**2)
            print("D space = ", dspace)
        if(self.cmbCrystal.currentText() =='Ge'):
            dspace = 5.658 / np.sqrt(int(refl[0])**2 + int(refl[1])**2 + int(refl[2])**2)
        if(self.cmbCrystal.currentText() =='SiO2'):
            dspace = 4.564 / 2
#            a=4.91304
#            c=5.40463
#            dspace = 5.4307 / np.sqrt(int(refl[0])**2 + int(refl[1])**2 + int(refl[2])**2)
        
        
        self.calibrationWaveLength = list(1239.84187 / np.array(self.calibrationEnergies))
        
        self.calibrationAngles = list(np.degrees(np.arcsin(self.calibrationWaveLength / (2*dspace/10))))
        
        self.lstEnergies.clear()
        self.lstPositions.clear()
        
        self.lstEnergies.addItems((["%.3f" % x for x in self.calibrationEnergies]))
#        for i in range(len(self.calibrationEnergies)):
#            self.lstPositions.addItems(str(self.calibrationPositions))
            
        self.lstEnergies.setCurrentRow(0)
        self.lstEnergiesItemClicked()
        
#        #depending on the crystal and orientation used calculate d spacing
#        refl = list(self.edtCrystalReflection.text()) 
#        if(self.cmbCrystal.currentText() == 'Si'):
#            dspace = 5.4307 / np.sqrt(int(refl[0])**2 + int(refl[1])**2 + int(refl[2])**2)
#            print("D space = ", dspace)
#        if(self.cmbCrystal.currentText() == 'Ge'):
#            dspace = 5.658 / np.sqrt(int(refl[0])**2 + int(refl[1])**2 + int(refl[2])**2)
#
#        #energy info available from hdf5 file
#        try:      self.calibrationEnergies.append(self.current_img.energy)
#        except:   self.calibrationEnergies.append(0)
#        
#        try:      self.calibrationWaveLength.append(1239.84187 / self.current_img.energy)
#        except:   self.calibrationWaveLength.append(0)
#        
#        try:      self.calibrationAngles.append(np.degrees(np.arcsin(self.calibrationWaveLength[-1] / (2*dspace/10))))
#        except:   self.calibrationAngles.append(0)
#
#        self.calibrationPositions.append(positions)
#        self.lstEnergies.addItem(str(self.calibrationEnergies[-1]))
#        self.lstPositions.clear()
#        self.lstPositions.addItems(["%.3f" % x for x in self.calibrationPositions[-1]])
#        self.canv.draw()
    
    def lstPositionsItemDoubleClicked(self):
        item = self.lstPositions.currentItem()
        self.old_val = item.text()
        item.setFlags(item.flags() | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled)
        self.lstPositions.editItem(self.lstPositions.currentItem())
        
    def lstPositionsItemChanged(self):
        if self.old_val != self.lstPositions.currentItem().text():
            itemNr = self.lstEnergies.currentRow()
            cntPos = self.lstPositions.currentRow()
            self.calibrationPositions[itemNr][cntPos] = float(self.lstPositions.currentItem().text())
            self.lstEnergiesItemClicked()
            print("changed")
            
    def lstPositionsItemClicked(self):
        
        itemNrE = self.lstEnergies.currentRow()
        itemNrP = self.lstPositions.currentRow()
        
        pos = self.calibrationPositions[itemNrE][itemNrP]
        
        for line in self.ax_image.lines:
            if line.get_color() == 'r':
                line.set_color('k')
        
        self.ax_image.axvline(x=pos, linewidth=1, color = 'r')
        self.canv.draw()
    
    def lstEnergiesItemClicked(self):
        itemNr = self.lstEnergies.currentRow()
        self.lstPositions.clear()
        if self.canv.draw() != []:
            for i in range(len(self.ax_image.lines)):
                self.ax_image.lines.pop(0)
        
        try:
            if len(self.calibrationPositions[itemNr]) != 0:
                self.lstPositions.addItems(["%.3f" % x for x in self.calibrationPositions[itemNr]])
    
                for pos in self.calibrationPositions[itemNr]:
                    self.ax_image.axvline(x=pos, linewidth=1, color = 'k')
                    
                self.canv.draw()
        except: return
    
    def lstEnergiesItemDoubleClicked(self):
        item = self.lstEnergies.currentItem()
        self.old_val = item.text()
        item.setFlags(item.flags() | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled)
        self.lstEnergies.editItem(self.lstEnergies.currentItem())
        
    def lstEnergiesItemChanged(self):
        if self.old_val != self.lstEnergies.currentItem().text():
            itemNr = self.lstEnergies.currentRow()
            self.calibrationEnergies[itemNr] = float(self.lstEnergies.currentItem().text())
            self.calibrationWaveLength[itemNr] = 1239.84187 / self.calibrationEnergies[itemNr]

            #depending on the crystal and orientation used calculate d spacing
            refl = list(self.edtCrystalReflection.text()) 
            if(self.cmbCrystal.currentText() =='Si'):
                dspace = 5.4307 / np.sqrt(int(refl[0])**2 + int(refl[1])**2 + int(refl[2])**2)
                print("D space = ", dspace)
            if(self.cmbCrystal.currentText() =='Ge'):
                dspace = 5.658 / np.sqrt(int(refl[0])**2 + int(refl[1])**2 + int(refl[2])**2)
            if(self.cmbCrystal.currentText() =='SiO2'):
                dspace = 4.564 / 2
    #            a=4.91304
    #            c=5.40463
    #            dspace = 5.4307 / np.sqrt(int(refl[0])**2 + int(refl[1])**2 + int(refl[2])**2)

            self.calibrationAngles[itemNr] = np.degrees(np.arcsin(self.calibrationWaveLength[itemNr] / (2*dspace/10)))
    
    def lstRegionsItemClicked(self): 
        index = self.lstRegions.currentRow()
        if self.lstAngles.currentRow() != index:
            self.lstAngles.setCurrentRow(index)
            self.lstAnglesItemClicked()
        self.plotROI()
     
    def lstRegionsItemDoubleClicked(self):
        item = self.lstRegions.currentItem()
        self.old_region = item.text()
        item.setFlags(item.flags() | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled)
        self.lstRegions.editItem(self.lstRegions.currentItem())
        
    def lstRegionsItemChanged(self):
        if self.old_region != self.lstRegions.currentItem().text():
            index = self.lstRegions.currentRow()
            try:
                string = self.lstRegions.currentItem().text()
                string = string.strip()
                reglist = string.split(',')
                reg_start = int(reglist[0])
                reg_end = int(reglist[1])
            except:
                print("Could not retrieve region indices. Please check input.")
                self.lstRegions.currentItem().setText(self.old_region)
                return
            
            self.calibrate_parms['region_inds'][index] = [reg_start, reg_end] 
            self.plotRegionBorders()
            
    def lstAnglesItemClicked(self): 
        index = self.lstAngles.currentRow()
        if self.lstRegions.currentRow() != index:
            self.lstRegions.setCurrentRow(index)
            self.lstRegionsItemClicked()
        pass
     
    def lstAnglesItemDoubleClicked(self):
        item = self.lstAngles.currentItem()
        self.old_angle = item.text()
        item.setFlags(item.flags() | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled)
        self.lstRegions.editItem(self.lstAngles.currentItem())
        
    def lstAnglesItemChanged(self):
        if self.old_angle != self.lstAngles.currentItem().text():
            index = self.lstAngles.currentRow()
            try:
                angle = float(self.lstAngles.currentItem().text())
            except:
                print("Could not retrieve new angle. Please check input.")
                self.lstRegions.currentItem().setText(self.old_angle)
                return
            
            self.calibrate_parms['angles'][index] = angle
        
            self.plotROI()
        
        
    def closeEvent(self, event):
        self.saveSettings()
        super(calibrationWidget, self).closeEvent(event)
        print('closed')

    def saveSettings(self):
        settings = QtCore.QSettings('myorg', 'calibrationWidget')
        settings.setValue('emin', self.edtEmin.text())
        settings.setValue('emax', self.edtEmax.text())
        settings.setValue('de', self.edtdE.text())
        settings.setValue('reflection', self.edtCrystalReflection.text())
        settings.setValue('crystal', str(self.cmbCrystal.currentIndex()))
        

    def restoreSettings(self):
        settings = QtCore.QSettings('myorg', 'calibrationWidget')
        self.edtdE.setText(str(settings.value('de')))
        self.edtCrystalReflection.setText(str(settings.value('reflection')))
        try:
            self.cmbCrystal.setCurrentIndex(int(str(settings.value('crystal'))))
        except:
            self.cmbCrystal.setCurrentIndex(0)
            
    def imageDoubleClick(self, event):
        print(event)
        if event.dblclick == True:
            #ask to change scale
            num, ok = QtGui.QInputDialog.getDouble(self,
                                               "Z scale",
                                               "Z max",
                                               value = self.zmax, 
                                               max = 200000,
                                               min = 0,
                                               decimals = 10)
            if ok:
                self.zmax = num
                self.image.set_clim(vmax=self.zmax)
                self.canv.draw()
            else:
                return
            
    def recalcAngles(self):
        #depending on the crystal and orientation used calculate d spacing
        refl = self.edtCrystalReflection.text()
        if len(refl)>3:
            refl = refl.split(',')
            print(refl)
        else:
            refl = list(self.edtCrystalReflection.text())
        if(self.cmbCrystal.currentText() =='Si'):
            dspace = 5.4307 / np.sqrt(int(refl[0])**2 + int(refl[1])**2 + int(refl[2])**2)
            print("Si D space = ", dspace)
        if(self.cmbCrystal.currentText() =='Ge'):
            dspace = 5.658 / np.sqrt(int(refl[0])**2 + int(refl[1])**2 + int(refl[2])**2)
            print("Ge D space = ", dspace)
        if(self.cmbCrystal.currentText() =='SiO2'):
            dspace = 4.564 / 2
#            a=4.91304
#            c=5.40463
#            dspace = 5.4307 / np.sqrt(int(refl[0])**2 + int(refl[1])**2 + int(refl[2])**2)
        
        
        self.calibrationWaveLength = list(1239.84187 / np.array(self.calibrationEnergies))
        
        self.calibrationAngles = list(np.degrees(np.arcsin(self.calibrationWaveLength / (2*dspace/10))))
        
    def calculateXanes(self):
        #TODO not tested!
        cnt = self.lstCalibrationFiles.currentRow()
        xanes = []
        xaness = []
        xanesE = []
        xanesRegion = self.edtregionXanes.text().split()
        eminXanes = float(xanesRegion[0])
        emaxXanes = float(xanesRegion[1])
        
        emin = float(self.edtEmin.text())
        emax = float(self.edtEmax.text())
        de = float(self.edtdE.text())
        
        #depending on the crystal and orientation used calculate d spacing
        refl = list(self.edtCrystalReflection.text()) 
        if(self.cmbCrystal.currentText() =='Si'):
            dspace = 5.4307 / np.sqrt(int(refl[0])**2 + int(refl[1])**2 + int(refl[2])**2)
            print("D space = ", dspace)
        if(self.cmbCrystal.currentText() =='Ge'):
            dspace = 5.658 / np.sqrt(int(refl[0])**2 + int(refl[1])**2 + int(refl[2])**2)
        if(self.cmbCrystal.currentText() =='SiO2'):
            dspace = 4.564 / 2
#            a=4.91304
#            c=5.40463
#            dspace = 5.4307 / np.sqrt(int(refl[0])**2 + int(refl[1])**2 + int(refl[2])**2)
        
        self.calibrationCoeficients = []
        


        for k in range(len(self.np_data)):
            forAveraging = []
            commonE = []
            columns = np.arange(0, len(self.np_data[0][0]), 1)
    #        self.ax_calibratedImage.clear()
    
            self.calibrationCoeficients = []
            for i in range(len(self.calibrationPositions[0])):
                posit = []
                ang = []
                for j in range(len(self.calibrationEnergies)):
                    if self.calibrationPositions[j] != []:
                        posit.append(self.calibrationPositions[j][i])
                        ang.append(self.calibrationAngles[j])
    
                a, b = np.polyfit(posit, ang, 1)
                self.calibrationCoeficients.append([a,b])

                
                angle_scale = columns*self.calibrationCoeficients[-1][0]+ self.calibrationCoeficients[-1][1]
                print("Angle scale ", angle_scale[0], angle_scale[-1])
    
                enscale = 1239.84187 / ((2*dspace/10) * np.sin(np.radians(angle_scale)))
                print("Energy scale ", enscale[0], enscale[-1])
        
                xmax = len(self.np_data[0])
    #            roi = self.np_data[0:xmax, int(self.calibrate_parms['region_inds'][i][2]):int(self.calibrate_parms['region_inds'][i][3])]
                roi = np.array([self.np_data[k]])[0:xmax, int(self.calibrate_parms['region_inds'][i][2]):int(self.calibrate_parms['region_inds'][i][3])]
                vertSum = np.sum(roi, axis=1)

                commonE, rdata = rebin_data(enscale, vertSum[0], emin, emax, de)
                forAveraging.append(rdata)            
                        
                average =  np.sum(forAveraging, axis=0) / len(forAveraging)

            roiXanes = np.where( (commonE>eminXanes) & (commonE<emaxXanes) )
            
            if(XANES_SMOOTH):
                spl = Rbf( commonE[roiXanes], average[roiXanes], 
                           function = 'multiquadric', 
                           epsilon = 3, 
                           smooth = 0.1 )
                
                averages = spl(commonE[roiXanes])
                xanesPoints = np.sum( averages ) / self.i0[k]
                xaness.append(xanesPoints)

            xanesPoint = np.sum( average[roiXanes] ) / self.i0[k]
                
            xanesE.append(self.energy[k])
            xanes.append(xanesPoint)
            

        self.ax_xanes.plot(xanesE, xanes)
        self.ax_xanes.plot(xanesE, xaness)
        self.ax_xanes.relim()
        self.ax_xanes.autoscale()
        self.canv3.show()
        
        cnt = self.lstCalibrationFiles.currentRow()
        savefile = self.keys[cnt] + '.xanes'
        forSave = []
        forSave.append(xanesE)
        forSave.append(xanes)
#        forSave = np.concatenate((forSave, forAveraging))
        np.savetxt(savefile, np.transpose(forSave), header = "Energy xanes")


            
            
class TestWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(TestWindow, self).__init__()  
        self.setWindowTitle("CalibrationWidget v9")
        
        print(os.getpid())
        
        self.wid = calibrationWidget(self)
        self.setCentralWidget(self.wid) 

        self.show()
        
    def closeEvent(self, event):
        self.wid.saveSettings()
        super(TestWindow, self).closeEvent(event)
        print('closed')
        
if __name__ == '__main__':
    app = QtGui.QApplication(argv)

    main = TestWindow()

    exit(app.exec_())
    
    pass
            
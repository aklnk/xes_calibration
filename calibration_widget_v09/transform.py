# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 15:53:13 2018

@author: angerpos
"""

import numpy as np

from PIL import Image
from sklearn import preprocessing
from scipy import optimize
from scipy.ndimage.interpolation import rotate

from peak_detection import peaks_detection


def gauss(x, a, x_0, sigma, offset):
    '''
    Compute Gaussian at position x.
    
    Output: float
    '''        
    return (a * np.exp(-(x-x_0)**2 / (2 * sigma**2))) + offset

def rescale(image, factor):
    '''
    Rescale image by a constant factor (I0 correction).
    
    Output: rescaled image
    '''    
    return image / factor

def get_scaler(image):
    '''
    Determine min-max scaling from image to apply to all.
    Image needs to be bad-pixel corrected!
    
    Output: preprocessing.MinMaxScaler object
    '''
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(image)
    
    return scaler

def normalize(image, scaler):
    '''
    Apply scaler to to image (e.g. min-max scaler).
    
    Output: rescaled image
    '''
    return scaler.transform(image)

def flat_correct(image, flat_file):
    '''
    Divide image by flat field image (component-wise).
    
    Output: rescaled image
    '''        
    flat_image = np.array(Image.open(flat_file))
    
    #Handle error for mismatch in shapes?
    return image / flat_image    

def black_mask(black_file):
    '''
    Determine bad pixel mask from black image measurement.
    
    Output: mask (bool type array_like image)
    '''
    #load black image (tif)
    black_image = np.array(Image.open(black_file))
    mask = np.zeros_like(black_image, dtype=np.bool)
    
    for index in np.argwhere(black_image > 0.0):
        mask[tuple(index)] = True
    
    return mask

def gradient_mask_old(image, scale=5.0):
    '''
    Determine bad pixel mask from image by looking at the gradient.
    
    Output: mask (bool type array_like image)
    ''' 
    #define noise level
    noise = max_noise(image)   
    flat = image.flatten()
    #create mask
    mask = np.zeros_like(flat, dtype=np.bool)        
    
    for i in range(len(flat)):
        #set dynamic threshold and pixel value (pseudo-black/white)
        if flat[i] <= noise:
            threshold = (scale)*noise
            pixel = noise
        else:
            threshold = scale*flat[i]
            pixel = flat[i]
            
        #trigger bad pixel by looking at gradient            
        if (not mask[i]) and (flat[(i+1)%len(flat)] - pixel >= threshold):
            #scale too high??
            #store previous intensity level as reference
            current = np.average(flat[i-2:i+1])*scale
            #mark next pixel as bad
            mask[(i+1)%len(flat)] = True
        #continue marking bad pixels if value is above reference
        elif mask[i] and flat[(i+1)%len(flat)] > current:
            mask[(i+1)%len(flat)] = True
            continue

    return mask.reshape(image.shape)

def gradient_mask(image, scale=5.0):
    '''
    Determine bad pixel mask from image by looking at the gradient.
    
    Output: mask (bool type array_like image)
    ''' 
    #define noise level
    noise = max_noise(image)   
    flat = image.flatten()
    #create mask
    mask = np.zeros_like(flat, dtype=np.bool)    
    #create threshold
    current = np.ones_like(flat) * noise
    thresh_mask = np.argwhere(flat > noise)
    for index in thresh_mask:
        current[tuple(index)] = flat[tuple(index)]
    threshold = current * scale
    #threshold[np.argwhere(flat > noise)] = 
    #create gradient
    grad = np.diff(current)
    
    for i in range(len(grad)):
            
        #trigger bad pixel by looking at gradient            
        if (not mask[i]) and (grad[i] >= threshold[i]):
            #scale too high??
            #store previous intensity level as reference
            good_thresh = threshold[i]
            #mark next pixel as bad
            mask[(i+1)%len(flat)] = True
        #continue marking bad pixels if value is above reference
        elif mask[i] and flat[(i+1)%len(flat)] > good_thresh:
            mask[(i+1)%len(flat)] = True
            #continue

    return mask.reshape(image.shape)

def merge_masks(masks):
    '''
    Merge to masks by element-wise OR operation.
    
    Output: mask (bool type array_like image)
    '''         
    merge = np.vectorize(lambda a, b: a or b)
    current = masks.pop()
    while masks:
        current = merge(current, masks.pop())
    
    return current

def apply_mask(image, mask):
    '''
    Apply mask to image and average over masked pixels.
    
    Output: None (operates directly on input)
    '''         
    for index in np.argwhere(mask):
        average_neighbors(image, mask, index)
    
    return


def chunk_avg(image, xsec=10, ysec=8):#, chunks=10*10):
    '''
    Get average and std. dev. from chunks (split image in xsec * ysec)
    e.g. to determine approximation to noise level
    
    Output: arrays (average,  st.dev.,  chunks) with shape (ysec, xsec)
    '''
    chunks = []
    avg = []
    std = []
    #Split image in xsec*ysec chunks
    for strip in np.array_split(image, ysec):
        for chunk in np.array_split(strip, xsec, axis=1):
            chunks.append(chunk)
            avg.append(np.average(chunk))
            std.append(np.std(chunk))
    try:
        chunks = np.array(chunks).reshape((ysec, xsec))
    except:
        ysec=ysec-1
        chunks = []
        avg = []
        std = []
        #Split image in xsec*ysec chunks
        for strip in np.array_split(image, ysec):
            for chunk in np.array_split(strip, xsec, axis=1):
                chunks.append(chunk)
                avg.append(np.average(chunk))
                std.append(np.std(chunk))      
#    for i in range(len(chunks)):
#        print(chunks[i].shape)
    chunks = np.array(chunks).reshape((ysec, xsec))
#    chunks = np.array(chunks)
    
    avg = np.array(avg).reshape((ysec, xsec))
    std = np.array(std).reshape((ysec, xsec))

    return avg, std, chunks

def max_noise(image):
    '''
    Determine estimate for maximal noise intensity.
    
    Output: float
    '''    
    chunks = chunk_avg(image, xsec=10, ysec=5)
    minnoise = np.min(chunks[0])
    minchunk = chunks[2][tuple(np.argwhere(chunks[0] == minnoise)[0])]
    out = np.max(minchunk)    
    return out

def std_noise(image):
    '''
    Determine standard deviation of noise (analog to max_noise) 
    
    Output: float
    '''    
    chunks = chunk_avg(image, xsec=10, ysec=5)
    minnoise = np.min(chunks[0])
    minchunk = chunks[2][tuple(np.argwhere(chunks[0] == minnoise)[0])]
    out = np.std(minchunk)       
    
    return out

def average_neighbors(image, mask, index, radius=1):
    '''
    Iterate over marked pixels and average over closest correct pixels.
    
    Output: None (operates directly on input)
    '''       
    #mask contains positions of bad pixels
    if not mask[tuple(index)]:
        return
    #unit directions to find neighbors
    x_, y_ = np.eye(2, dtype=np.int)
    nghbr = []
    #iterate over [-r,  -r+1,  ...,  r]**2 around pixel
    for x in np.arange(-radius, radius+1):
        for y in np.arange(-radius, radius+1):
            #neighbor index
            ind_new = index + x*x_ + y*y_
            #avoid stepping out of the image
            try:
                #ignore bad pixels when averaging
                if mask[tuple(ind_new)]:
                    continue
                #savevalue for averaging
                else:
                    nghbr.append(image[tuple(ind_new)])
            except IndexError:
                print("Warning: stepped over edge while trying to average pixel", x, y)
                continue
    #increase radius iteratively if only bad pixels around
    if nghbr == []:
        return average_neighbors(image, mask, index, radius+1)
    #replace bad pixel by average
    else:
        avg = np.average(nghbr)
        image[tuple(index)] = avg
    
    return    

def cut_noise_old(image, noise_lvl, b_w=True):
    '''
    Reduce all pixels below noise level to 0 and (if bw) others to 1.
    
    Output: b/w image
    '''    
    flat = image.flatten()
    
    for i in range(len(flat)):
        if flat[i] <= noise_lvl:
            flat[i] = np.float64(0.)
        elif b_w:
            flat[i] = np.float64(1.)
    
    return flat.reshape(image.shape)

def cut_noise(image, noise_lvl, b_w=True):
    '''
    Reduce all pixels below noise level to 0 and (if bw) others to 1.
    
    Output: b/w image
    '''    
    flat = image.flatten()
    out = np.zeros_like(flat)
    mask = np.argwhere(flat > noise_lvl)
    
    for index in mask:
        if b_w:
            out[tuple(index)] = np.float64(1.)
        else:
            out[tuple(index)] = flat[tuple(index)]

    return out.reshape(image.shape)

def project_image(image, axis, wght=None, mode="sum"):
    '''
    Project intensity to given axis (avg or sum)
        x-axis: 1
        y-axis: 0
        
    Output: 1D projection
    '''
    if mode == "sum":
        return np.sum(image, axis=axis, dtype=np.float64)
    elif mode == "avg":
        return np.average(image, axis=axis, weights=wght)

def best_peaks(data, num, bounds=None, scales=np.arange(1, 30), it=20, dist=10):
    '''
    Find "num" peaks in data
    (1D array,  e.g. projection of intensity to y axis)
    
    Output: peak-indices,  peak-signals (tuples)
    '''
    #define bounds for search (to restrict to certain areas of the image)
    if bounds == None:
        bounds = [0, len(data)]
    #add epsilon to data to avoid problems with wavelet tranform (no zeros!)
    datanew = data + 1e-5
    #remove uninteresting area from peak search
    datanew[0:bounds[0]] = 1e-5
    datanew[bounds[1]:len(data)] = 1e-5
    #shift points liearly by epsilon to avoide "peak plateau"
    for i in range(len(datanew)):
        datanew[i] += i * 1e-7
    #cap no. of iterations in while loop        
    i = 0
    #signal to noise ratio - start with O(max of data)
    snr = np.max(data) * 0.5
    #peak-finder
    peaks, sig = peaks_detection(datanew, scales, snr)
    #decrease snr until enough peaks are found
    while len(peaks) < num * 2.0:
        snr = snr * 0.75
        peaks, sig = peaks_detection(datanew, scales, snr)
        i += 1
        if i > it:
            print("Warning: number of allowed iterations in peak finding exceeded")
            break
    #return first "num" peaks,  ordered by signal intensity
    try:
        #check if peaks are far enough apart and keep higher peak if not
        ind = []    
        for i in range(len(peaks)-1):        
            if peaks[i+1] - peaks[i] < dist:
                ind.append(int(np.where(sig[i] >= sig[i+1], i+1, i)))
        peaks = list(np.delete(peaks, ind, 0))
        sig = list(np.delete(sig, ind, 0))
        #return peaks if still enough found,  take num starting from highes sig
        sig, peaks = zip(*sorted(zip(sig, peaks), reverse=True))
        return peaks[0:num], sig[0:num]

    #still not enough peaks
    except:
        print("Warning: not enough peaks found")
        return peaks, sig

def find_regions(data, peaks, thresh_scale=20.0):
    '''
    Find regions around peaks in data
    (1D array,  e.g. projection of intensity to y axis)
    Needs b/w data!
    
    Output: region indices (start, end) (list of lists)
    '''
    #Set threshold for noise level
    #thrshld = thresh_scale*np.average(data)
    # !!! Test
    weights = [np.exp(- (1/thresh_scale) * i) for i in data]
    thrshld = np.average(data, weights=weights)
    ####
    
    regions = []
    peaks = list(peaks)
    peaks.sort()
    
    print(peaks)
    
    #iterate over peaks
    for ind in peaks:
        #width of region to left/right
        left = 1
        right = 1
        try:
            #step left and check when noise level is reached
            while data[ind-left] > thrshld:
                left += 1
        except IndexError:
            print("Warning: stepped out of image while searching ROIs")
            left = ind
        try:
            while data[ind+right] > thrshld:
                right += 1
        except IndexError:
            print("Warning: stepped out of image while searching ROIs")
            right = len(data) - ind - 1    
        #output list of region as start, end along y-axis
        regions.append([ind-left, ind+right])
    #Check if regions overlap and set borders to average
    for i in range(len(regions)-1):
        if regions[i][1] > regions[i+1][0]:
            avg = np.round(np.average([regions[i][1], regions[i+1][0]])).astype(int)
            regions[i][1] = avg
            regions[i+1][0] = avg
    #sort regions from top to bottom of image
    regions.sort(key=lambda x: x[0])
    return regions

def max_snr(image, region):
    '''
    Compute maximal row-wise signal-to-noise ratio in given region.
    Used for finding optimal rotation angle for ROIs.
    
    Input: whole image data and slice of data (region)
    
    Output: float 
    '''
    #compute reference st. dev. of noise
    std = std_noise(image)
    #compute row-wise snr in the original part of the rotated region
    snr_region = np.array([np.mean(row)/std for row in region])
    
    return np.max(snr_region)

def rotate_region_old(image, region_inds, angle):
    '''
    Rotate region by given angle. Take buffer into account to reduce zeros in
    rotated image
    
    Output: rotated region as 2D array (cut out of data)
    '''
    #Compute buffer for height to avoid empty data in rotated section
    delta = ((region_inds[1] - region_inds[0])
             * np.cos(np.deg2rad(np.absolute(angle)))
             + image.shape[1] 
             * np.sin(np.deg2rad(np.absolute(angle))))
    delta = int((delta - (region_inds[1] - region_inds[0])) // 2)
    #actual region to rotate
    up = region_inds[0]-delta
    down = region_inds[1]+delta
    #check if region lies out of image and correct
    if up < 0:
        up = 0
    if down > image.shape[0]:
        down = image.shape[0]
    #cut region out of data and rotate
    strip = image[up:down]
    
    return rotate(strip, angle, reshape=False)[delta:len(strip)-delta] 

def rotate_region(image, region_inds, angle, delta_x=0):
    '''
    Rotate region by given angle. Take buffer into account to reduce zeros in
    rotated image
    
    Output: rotated region as 2D array (cut out of data)
    '''
    #Compute buffer for height to avoid empty data in rotated section
    height = (region_inds[1] - region_inds[0])
    width = (image.shape[1] + np.absolute(delta_x))
    
    delta = (height * np.cos(np.deg2rad(np.absolute(angle)))
            + width * np.sin(np.deg2rad(np.absolute(angle))))
    delta = int((delta - (region_inds[1] - region_inds[0])) // 2)
    #actual region to rotate
    up = region_inds[0]-delta
    down = region_inds[1]+delta

    pad_y = [0, 0]
    #check if region out of image and correct
    if up < 0:
        #print("up", up)
        pad_y[0] = - up
        up = 0
    if down > image.shape[0]:
        #print("down",down)
        pad_y[1] = down - image.shape[0]
        #down = data.shape[0]
    #cut region out of data and rotate
    strip = np.pad(image, [pad_y, [0,0]], 'reflect')[up:up + 2*delta + (region_inds[1]-region_inds[0])]

    pad_x = [0, 0]
   
    pad_x[int((np.sign(delta_x) +1)/2)] = 2 * np.absolute(delta_x)
    
    pads = [[0, 0], pad_x]
    
    strip = np.pad(strip, pads, mode='constant')
    return rotate(strip, angle, reshape=False)[delta:len(strip)-delta, pad_x[0]: strip.shape[1] - pad_x[1]]
    
    #return rot[delta:len(strip)-delta, pad_x[0]: strip.shape[1] - pad_x[1]]

def find_angle(image, region_inds, a_0=0.0, bnds=[(-10.0, 10.0)], method='SLSQP',
                  options = {'maxiter': 100, 'ftol': 1e-06, 'iprint': 1, 'disp': False, 'eps': 5.0e-02}):
    '''
    Find optimal angle to rotate region (i.e. the one which yields maximal
    signal-to-noise ratio)
    
    Needs bad pixel correction first!
    
    Output: angle (float)
    '''
    max_x = int(best_peaks(project_image(image[region_inds[0]:region_inds[1]], axis=0, mode="avg"),1)[0][0])
    del_x = max_x - int(image.shape[1]/2)
    
    #minimize (negative!) maximal snr in region
    f = lambda a: - max_snr(image, rotate_region(image, region_inds, a, delta_x=del_x))
    #scipy minimization. May need tweaking
    result = optimize.minimize(f, a_0, method=method, bounds=bnds, 
                               options=options)
    #rotation threshold (angle at which at least one pixel is rotated)
    #using sin(a) ~ a for a << 1
    threshold = np.rad2deg(2.0/image.shape[1])
    
    if np.absolute(result.x - a_0) <= threshold:
        return a_0, del_x
    else:
        return float(result.x), del_x

def find_elastic_position(image, region_inds, angle, refine=True, reg_kwargs={}, peak_kwargs={}):
    '''
    Find pixel position of elastic line for calibration.
    
    Output: position (int)
    '''
    
    #cut out region, rotate around angle and project to x axis (b/w)
    region = rotate_region(image, region_inds, angle)
    b_w_x = project_image(cut_noise(region, max_noise(image)), axis=0)
    #find single best peak
    peaks, sig = best_peaks(b_w_x, 1, **peak_kwargs)
    position = peaks[0]
    #Diagnostics
#    import matplotlib.pyplot as plt
#    plt.figure(dpi=200)
#    plt.plot(b_w_x, linewidth=0.6)
#    plt.tick_params(left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
#    plt.xlabel("pixel", fontdict=None, labelpad=None)
#    plt.ylabel("intensity", fontdict=None, labelpad=None)
#    plt.savefig('elasti_bw.png', transparent=True)
#    plt.show()

    if refine:
        region_x = project_image(region, axis=0, mode="avg")
        #TODO adjust this
        #region_x = np.exp(region_x)
        [refine_inds] = find_regions(b_w_x, peaks, **reg_kwargs)
        print(refine_inds)
#        try:
#            refine_inds[0] -= 10
#            refine_inds[1] += 10
#        except IndexError:
#            pass
        #initial guess for fitting
        if refine_inds[0]<0:
            refine_inds[0] = 0
        p_0 = [np.max(region_x[refine_inds[0]:refine_inds[1]]), 
               #position - refine_inds[0], 
               position,
               1.0, 
               np.min(chunk_avg(image, ysec=5)[0])]
        #fit Gaussian to data
        try:
            popt, pcov = optimize.curve_fit(gauss, 
                                        np.arange(refine_inds[0], refine_inds[1]), 
                                        region_x[refine_inds[0]:refine_inds[1]], 
                                        p_0)
        except:
            print("Can't refine position. Unrefined position used for region Nr.", region_inds)
            return position
        #Diagnostics
#        print("initial params: ", p_0)
#        print("fit params: ", popt)
#        
#        plt.figure(dpi=200)
#        plt.plot(np.arange(refine_inds[0], refine_inds[1]),region_x[refine_inds[0]:refine_inds[1]], linewidth=0.6)
#        plt.plot(np.arange(refine_inds[0], refine_inds[1]),np.vectorize(gauss)(np.arange(refine_inds[0], refine_inds[1]), *popt) , linewidth=0.6)
#        plt.tick_params(left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
#        plt.xlabel("pixel", fontdict=None, labelpad=None)
#        plt.ylabel("intensity", fontdict=None, labelpad=None)
#        plt.savefig('elasti_fit.png', transparent=True)
#        plt.show()
        
        #check if refined position is still inside region and if fit has indeed a maximum
        if int(popt[1]) in range(refine_inds[0], refine_inds[1]) and popt[0] > 0:
            position = popt[1]
            
    return position

def find_roi(image, num):
    '''
    Find ROIs from image, combining previous methods.
    
    Output: region indices (start, end) (list of lists)
    '''
    b_w_y = project_image(cut_noise(image, max_noise(image)), axis=1)
    
    peaks, sig = best_peaks(b_w_y, num)
    
    return find_regions(b_w_y, peaks)
    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import collections as col

    black_file = "FullModule9_4keV_mask.tif"
    flat_file = "FullModule9_4keV_Cuflatfield.tif"
#    
#    dic = col.OrderedDict()
#    dic["a"] = [True, 1.0]
#    dic["b"] = [False, 2.3]
#    print(dic)
#    
#    for key in dic:
#        print(dic[key])
    
    import h5py
    import time
    
    NAME_OLD = "lambda_images"
    file = h5py.File("D:/expdata/2017-12-Chaturvedi/xes_data/S10_7140eV_300K_0004.hdf5", "r")
    file = h5py.File("D:/expdata/xes/Cage/Cage_pellet_fe_kbeta_15min_pX_7140eV0004.hdf5", "r")
    file = h5py.File("D:/expdata/xes/Cage/Cage_pellet_fe_kbeta_15min_pX_7040eV0001.hdf5", "r")
    #file = h5py.File("D:/expdata/2017-12-Chaturvedi/rixs_data/S4rixs_7122eV_300K_0003.hdf5", "r")
    
    
    data0 = np.array(file.get(NAME_OLD))[0]
    msk = gradient_mask(data0)
    apply_mask(data0, msk)
    
    data1 = np.copy(data0)
    apply_mask(data1,msk)
    
    roi = find_roi(data0,8)
    print(roi)    
    
    plt.figure(dpi=500)
    plt.imshow(data0[roi[-1][0]:roi[-1][1]], cmap='gist_ncar', vmax=150)
    #plt.imshow(mask, cmap='gray_r', vmax=1, alpha=0.5)
    plt.axis('off')
    plt.savefig('demo.png', transparent=True)
    plt.show()
    
    pos0 = find_elastic_position(data0, roi[-1], 0.0)
    
#    plt.figure(dpi=300)
#    #plt.plot(np.flip(nbw_y, 0), linewidth=0.4)
#    plt.plot(, linewidth=0.4)
#    plt.tick_params(left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
#    plt.xlabel("pixel", fontdict=None, labelpad=None)
#    plt.ylabel("intensity", fontdict=None, labelpad=None)
#    plt.savefig('demo3.png', transparent=True)
#    plt.show()
    
#    img_roi = np.loadtxt("test_chat_peak.csv", delimiter=",")
    
#    num = 10
#    noise = max_noise(data0)
#    
#    print("Start benchmark with {0} iterations...".format(num))
#    
#    t0 = time.time()
#    for _ in range(num):
#        #cut_noise2(data0, noise)
#        #mask1 = gradient_mask(data0)
#    
#    t1 = time.time()
#    for _ in range(num):
#        #cut_noise3(data0, noise)
#        #mask2 = gradient_mask2(data0)
#        
#    t2 = time.time()
#    
#    print("Old method takes {0} seconds.".format((t1-t0)/num))
#    print("New method takes {0} seconds.".format((t2-t1)/num))
#    
#    print(np.argwhere(mask1 != mask2))
    
#    img_cal_1 = np.loadtxt("img_cal_1.csv", delimiter=",") 
#    #img_cal_2 = np.loadtxt("img_cal_2.csv", delimiter=",")
#    
#    roi = find_roi(img_roi,8)
#    print(roi)
#    
#    angles = []
#    pos_1 = []
#    pos_2 = []
#    pos_1_ref = []
#    pos_2_ref = []
#    
#    msk_roi = gradient_mask(img_roi)
#    apply_mask(img_roi, msk_roi)
#    
#    for reg in roi:
#        angle = find_angle(img_roi, reg)
#        angles.append(angle)
#        pos_1.append(find_elastic_position(img_cal_1, reg, angle, refine=False))
#        #pos_2.append(find_elastic_position(img_cal_2, reg, angle, refine=False))
#        pos_1_ref.append(find_elastic_position(img_cal_1, reg, angle))
#        #pos_2_ref.append(find_elastic_position(img_cal_2, reg, angle))        
#
#    print(angles)
#    print(pos_1)
#    print(pos_1_ref)
#    #print(pos_2)
#    #print(pos_2_ref)    
#    
#    #img2 = flat_correct(img, flat_file)
#    
#    msk_cal_1 = gradient_mask(img_cal_1)
#    #msk_cal_2 = gradient_mask(img_cal_2)
#    apply_mask(img_cal_1, msk_cal_1)
#    #apply_mask(img_cal_2, msk_cal_2)
#    
#    #bw_y = project_image(cut_noise(img,max_noise(img)),1)
#    #peaks, sig = best_peaks(bw_y,8)
#    #regions = find_regions(bw_y,peaks)
#    #print(regions)
#    
#    #pos_1 = find_elastic_position(img,regions[1],0.0)
#    #print(pos_1)
#    
#    plt.figure(dpi=400)
#    plt.imshow(img_cal_1, cmap='gist_ncar')
#    for pos in pos_1:
#        plt.axvline(x=pos, color='r', linewidth=0.25)
#    for pos in pos_1_ref:
#        plt.axvline(x=pos, color='g', linewidth=0.25)
#    plt.show()
#    
##    plt.figure(dpi=400)
##    plt.imshow(img_cal_2, cmap='gist_ncar')
##    for pos in pos_2:
##        plt.axvline(x=pos, color='r', linewidth=0.25)
##    for pos in pos_2_ref:
##        plt.axvline(x=pos, color='g', linewidth=0.25)
##    plt.show()    
import os
import numpy as np  # NumPy (multidimensional arrays, linear algebra, ...)
import scipy as sp  # SciPy (signal and image processing library)

import matplotlib as mpl         # Matplotlib (2D/3D plotting library)
#mpl.use('Agg')
import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
#DEPRACATED from pylab import *              # Matplotlib's pylab interface

#from PIL import Image
import scipy.fftpack as ft
from scipy.optimize import leastsq as spleastsq
from scipy.optimize import minimize as spminimize
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mtick
from matplotlib.ticker import LogLocator
#from scipy.ndimage import correlate as ndcorrelate
#from scipy.ndimage import convolve as ndconvolve
#from scipy.signal import convolve2d, correlate2d
from scipy.interpolate import interp1d
from scipy.constants import k, u
from pypeaks import Data, Intervals
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage import gaussian_filter

from IPython.display import HTML

#from numba import jit

# Customisations
mpl.rcParams['mathtext.fontset'] = 'stix'
Formatter0 = mtick.ScalarFormatter(useMathText=True)

# Turn on Matplotlib's interactive mode - in pylab
#ion()


def display_embedded_video(filename):
    video = open(filename, "rb").read()
    video_encoded = video.encode("base64")
    video_tag = '<video controls alt="test" src="data:video/x-m4v;base64,{0}">'.format(video_encoded)
    return HTML(video_tag)

def gaussian_hwhm_to_radius(hwhm):
    return hwhm*np.sqrt(2/np.log(2))

def gaussian_radius_to_hwhm(radius):
    return radius*np.sqrt(np.log(2)/2)
    
def gaussian_radius_to_sig(radius):
    return radius / 2.

def gaussian_sig_to_radius(sig):
    return sig * 2.

def gaussian_hwhm_to_sig(hwhm):
    return hwhm/np.sqrt(2*np.log(2))

def gaussian_sig_to_hwhm(sig):
    return sig * np.sqrt(2*np.log(2))

#def gaussian1d(height, x0, hwhm, offset):
#    """Returns a function of the Gauss distribution for a given parameter set.\n
#    Example:\n
#    >>> g = gaussian1d(1,0,1.2,0.)\
#    x = np.linspace(0,2,100)\
#    plt.plot(x,g(x))"""
#    hwhm = float(hwhm)
#    return lambda x: height*np.exp(-1*((x-x0)/(hwhm))**2*np.log(2))\
#                    + offset

def gaussian1d(height, x0, sig, offset):
    """Returns a function of the Gauss distribution for a given parameter set.\n
    Example:\n
    >>> g = gaussian1d(1,0,1.2,0.)\
    x = np.linspace(0,2,100)\
    plt.plot(x,g(x))"""
    sig = float(sig)
    return lambda x: height*np.exp(-0.5*((x-x0)/sig)**2) + offset

#def residuals_gaussian1d(p,y,x):
#    """DEPRECATED\n
#    Calculates the array of residuals from Gaussian distribution"""
#    gmax, gx0, gfw, goffset = p
#    err = y - gmax*np.exp(-1*pow((x-gx0)/(gfw/2/np.log(2)),2)) - goffset
#    return err

def moments1d(x,data):
    """Returns (height, x0, stdev, offset) the gaussian parameters of 1D
    distribution found by a fit"""
    total = data.sum()
    if (x==None):
        x = np.arange(data.size)
    x0 = (x*data).sum()/total
    stdev = np.sqrt(np.sum((x-x0)**2*data)/np.sum(data))
    height = np.amax(data)
    offset = np.amin(data)
    return height, x0, stdev, offset

def fitgaussian1d(x,data):
    """Returns (height, centre, sigma, offset)
    the gaussian parameters of a 1D distribution found by a fit"""
    params = moments1d(x,data)
    if (x==None):
        errorfunction = lambda p: gaussian1d(*p)(*np.indices(data.shape))\
                                - data
    else:
        errorfunction = lambda p: gaussian1d(*p)(x)\
                                - data
    p, success = spleastsq(errorfunction, params, full_output=0)
    return p


def gaussian1d_no_offset(height, x0, sig):
    """Returns a function of the Gauss distribution for a given parameter set.\n
    Example:\n
    >>> g = gaussian1d(1,0,1.2,0.)\
    x = np.linspace(0,2,100)\
    plt.plot(x,g(x))"""
    sig = float(sig)
    return lambda x: height*np.exp(-0.5*((x-x0)/sig)**2)

def moments1d_no_offset(x,data):
    """Returns (height, x0, stdev, offset) the gaussian parameters of 1D
    distribution found by a fit"""
    total = data.sum()
    if (x==None):
        x = np.arange(data.size)
    x0 = (x*data).sum()/total
    stdev = np.sqrt(np.sum((x-x0)**2*data)/np.sum(data))
    height = np.amax(data)
    #offset = np.amin(data)
    return height, x0, stdev#, offset

def fitgaussian1d_no_offset(x,data):
    """Returns (height, centre, sigma, offset)
    the gaussian parameters of a 1D distribution found by a fit"""
    params = moments1d_no_offset(x,data)
    if (x==None):
        errorfunction = lambda p: gaussian1d_no_offset(*p)(*np.indices(data.shape))\
                                - data
    else:
        errorfunction = lambda p: gaussian1d_no_offset(*p)(x)\
                                - data
    p, success = spleastsq(errorfunction, params, full_output=0)
    return p
#def gaussian2d(height, x0, y0, hwhm_x, hwhm_y,offset):
#    """Returns a gaussian function with the given parameters"""
#    hwhm_x = float(hwhm_x)
#    hwhm_y = float(hwhm_y)
#    return lambda x,y: height*np.exp(
#                 -(((x-x0)/hwhm_x)**2+((y-y0)/hwhm_y)**2)*np.log(2))+offset


def gaussian2d(height, x0, y0, sig_x, sig_y,offset):
    """Returns a gaussian function with the given parameters:
    (height, y, x, sig_y, sig_x, offset)"""
    sig_x = float(sig_x)
    sig_y = float(sig_y)
    return lambda x,y: height*np.exp(
                 -0.5*(((x-x0)/sig_x)**2+((y-y0)/sig_y)**2))+offset


def moments2d(data):
    """Receives numpy array data with dim=2 and 
    returns (height, y, x, sig_y, sig_x, offset)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    sig_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    sig_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = np.nanmax(data)
    offset = np.nanmin(data)
    return height, x, y, sig_x, sig_y, offset


def fitgaussian2d(data):
    """Returns (height, y, x, sig_y, sig_x, offset)
    the gaussian parameters of a 2D distribution found by a fit"""
#    data = np.transpose(data)    
    params = moments2d(data)
    errorfunction = lambda p: np.ravel(gaussian2d(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = spleastsq(errorfunction, params, xtol=1e-16,ftol=1e-16)
    return p


def lorentz1d(height, x0, fwhm, offset):
    """Returns a function of the Lorentzian distribution for a given parameter set.\n
    Example:\n
    >>> lor = lorentz1d(1,0,1.2,0.)\
    x = np.linspace(0,2,100)\
    plt.plot(x,lor(x))"""
    fwhm = float(fwhm)    
    return lambda x: height/(1 + (2*(x-x0)/fwhm)**2) + offset

def residuals_lorentz(p,y,x):
    """DEPRECATED\n
    Calculates the array of residuals from Lorentz distribution"""
    lmax, lx0, lfw, loffset = p
    err = y - lmax/(1 + (2*(x-lx0)/lfw)**2) - loffset
    return err

def fitlorentz1d(x,data):
    """Returns (height, centre, fwhm, offset)
    the lorentzian parameters of a 1D distribution found by a fit"""
    params = moments1d(x,data)
    if (x==None):
        errorfunction = lambda p: lorentz1d(*p)(*np.indices(data.shape))\
                                - data
    else:
        errorfunction = lambda p: lorentz1d(*p)(x)\
                                - data        
    p, success = spleastsq(errorfunction, params, full_output=0)
    return p


def extinction_lorentz(b0,nu0,nu=1):
    """Function to calculate the transmission through a medium with a
    specific thickness. Receives b0, optical thickness at resonance,
    nu0, the offset from zero in centre frequency and offset in value
    from zero transmission, normally derived from laser linewidth."""
#    fwhm = float(fwhm)
#    nu = 6.066
    return lambda x: np.exp(-b0/(1 + (2*(x-nu0)/nu)**2))

def moments_extinction_lorentz(x,data):
    """Returns (b0, nu0, offset) the moments of transmission
    distribution for a fit"""
#    total = data.sum()
    if (x==None):
        x = np.arange(data.size)
    nu0 = x[np.argmin(data)]
    b0 = np.sqrt(((x-nu0)**2*data).sum()/data.sum())
#    offset = np.amin(data)
    return b0, nu0

def fit_extinction_lorentz(x,data):
    """Returns the (b0, nu0, offset) exponential decay with a lorentzian
    argument (b(nu)) parameters of a distribution found by a fit"""
    params = moments_extinction_lorentz(x,data)
#    params = np.array([8,0])
    if (x==None):
        errorfunction = lambda p: extinction_lorentz(*p)(*np.indices(data.shape))\
                                - data
    else:
        errorfunction = lambda p: extinction_lorentz(*p)(x)\
                                - data        
    p, success = spleastsq(func=errorfunction, x0=params)#, xtol=1e-16,ftol=1e-16)
    return p

    
def kinetic_expansion(Temp, sigma0):
    """Returns a function for time-of-flight measurements"""
    return lambda t: np.sqrt(Temp*t**2 + sigma0**2)

def moments_tof(x,data):
    """Calculates the initial parameters for a fit of time-of-flight to data"""
    sigma0 = np.amin(data)
    Temp = np.average((data**2-sigma0**2)/x**2)
    return Temp, sigma0

def fit_tof(x,data):
    """Returns the sigma0 and Temp for a time-of-flight measurements"""
    params = moments_tof(x,data)
    if (x==None):
        errorfunction = lambda p: kinetic_expansion(*p)(*np.indices(data.shape)) - data
    else:
        errorfunction = lambda p: kinetic_expansion(*p)(x) - data
    p, success = spleastsq(func=errorfunction, x0=params)#, xtol=1e-16,ftol=1e-16)
    return p

def decay_exp(N_0, tau):
    lambda_ = 1. / tau
#    return N_0 * np.exp(-lambda_ * t)
    return lambda t: N_0 * np.exp(-1 * lambda_ * t)

def moments_decay_exp(t,data):
    N_0 = np.amax(data)
    tau = t[len(t)/2]
    return N_0, tau

def fit_decay_exp(t,data):
    params = moments_decay_exp(t,data)
    errorfunction = lambda p: decay_exp(*p)(t) - data
    p = spleastsq(func=errorfunction, x0=params,
                  full_output=1)
    return p

def decay_logistic(N_0, steepness, shift):
    return lambda t: N_0 / (np.exp(steepness * (t - shift)) + 1)

def moments_decay_logistic(t,data):
    N_0 = np.amax(data)
    shift = t[len(t)/2]
    steepness = -4/N_0 * (data[len(t)/2 + 2] - data[len(t)/2 - 2]) / (t[len(t)/2 + 2] - t[len(t)/2 - 2])
    return N_0, steepness, shift

def fit_decay_logistic(t,data):
    params = moments_decay_logistic(t,data)
    errorfunction = lambda p: decay_logistic(*p)(t) - data
    p = spleastsq(func=errorfunction, x0=params,
                  full_output=1)
    return p

def chi_sq(y_data, y_fit, y_sigma):
    res = np.sum((y_data - y_fit)**2 / y_sigma**2)
    return res
    
def red_chi_sq(chi_sq, N, n):
    return chi_sq / (N - n - 1)

def low_pass_rfft(curve, low_freqs):
    """Filters the curve by setting to zero the high frequencies"""
    a = ft.rfft(curve)
    for i in range(2*low_freqs, len(curve)):
        a[i]=0    
    return np.array(ft.irfft(a))

def FourierFilter(function, half_interval):
    """Returns a fourier space filtered function by setting to zero all
    frequencies above half-interval and below -half-interval"""
    f_fft=ft.fft(function)
    for j in range(0,len(f_fft)):
        if(j>half_interval and j<len(f_fft)-half_interval):
            f_fft[j] = 0
    return ft.ifft(f_fft)



def prepare_for_fft_crop(input_image,fft_size,image_centre=0):
    """Returns an image cropped around image_centre with size fft_size.
    Image_centre should be a tuple with image centre coordinates
    or zero if centre should be found"""
    if image_centre != 0:
        centre_y, centre_x = image_centre
    else:
        x,y = input_image.shape
        centre_x = x/2
        centre_y = y/2
    if (x - centre_x < fft_size/2 or y - centre_y < fft_size/2):
        print "FFT size is bigger than the image itself!"
        return -1
    return input_image[centre_y-fft_size/2:centre_y+fft_size/2,\
        centre_x-fft_size/2:centre_x+fft_size/2]

def image_crop(input_image,ratio):
    """Returns a square image cropped around image_centre with ratio of initial
    image."""
    y,x = input_image.shape
    centre_x = x/2
    centre_y = y/2
    if x < y:
        new_size = int(x * ratio)
    else:
        new_size = int(y * ratio)
    return input_image[centre_y-new_size/2:centre_y+new_size/2,\
        centre_x-new_size/2:centre_x+new_size/2]

def prepare_for_fft_square_it(input_image):
    """Returns an image cropped around image_centre with square shape
    with the closest 2^n from below and padded with zeros if 2^n is
    smaller than 512"""
    y,x = input_image.shape
    a = np.amax([y,x])
    if y == a:
        cut = (y - x) / 2
        input_image = input_image[cut:x+cut,:]
    else:
        cut = (x - y) / 2
        input_image = input_image[:,cut:y+cut]
    #length = len(input_image)
    output_image = input_image
#    output_image = np.zeros((1024,1024))
#    padding = (1024 - length) / 2
#    output_image[padding:length+padding,padding:length+padding] = input_image
    return output_image

def prepare_for_fft_full_image(signal_image, gauss2D_param, gauss_sigma_frac):
    """Receives an input image and outputs the fourier transformed image
    and the cropped image used for the FFT. It has some Chameleon tunings"""
    param1 = gauss2D_param
    frac = gauss_sigma_frac
    centre1 = (param1[1],param1[2])
    dx1 = int(np.abs(param1[4]*frac))
    dy1 = int(np.abs(param1[3]*frac))
    
    #signal1 = np.array(plt.imread(signal_image),dtype=np.float64)
    #signal1 = signal1[1:]
    #if len(signal1.shape) > 2:
    #    signal1 = signal1[:,:,0]
    signal1 = signal_image[centre1[0]-dy1:centre1[0]+dy1, centre1[1]-dx1:centre1[1]+dx1]
    #signal1 = signal_image

    signal1 = prepare_for_fft_square_it(signal1)
    return signal1

def scattering_rate(I,delta):
    """Returns a fuction for calculating scattering rate (in MHz) for a beam of intensity
    I, transition with saturation parameter I_s=3.576 mW/cm^2 and detuning delta"""
    Gamma_sc= 0.5*38.11*(I/3.576)/(1 + (I/3.576) + 4*delta**2)
    return Gamma_sc

def circle_line_integration(image,radius):
    """Calculates the integral in a radial perimeter with input radius
    on input image and returns integral and pixels in integral"""
    if radius==0:
        return image[len(image)/2,len(image)/2], 1
#        return 0, 0
    if radius == 1:
        return image[len(image)/2-1:len(image)/2+2,len(image)/2-1:len(image)\
            /2+2].sum() - image[len(image)/2,len(image)/2], 8
    else:
        lx, ly = np.shape(image)
        x, y = np.ogrid[0:lx,0:ly]
        circle1 = (x-lx/2)**2 + (y-ly/2)**2 <= radius**2+1
        circle2 = (x-lx/2)**2 + (y-ly/2)**2 <= (radius-1)**2+1
#        image[circle1-circle2]=0
        return image[circle1-circle2].sum(), (circle1-circle2).sum()
    
def normalize_by_division(signal_image,ref_image):
    """Receives two images, the first one is the signal and the second
    the reference (e.g. gaussian spatial profile of a pump beam). 
    Divides the first by the second and removes any inf or nan in 
    the resultant matrix"""
    signal = signal_image / ref_image
    for pos in np.nditer(signal,op_flags=['readwrite']):
        if(pos==np.inf):
            pos[...] = 1.
        if(pos==-np.inf):
            pos[...] = 0
        if(pos < 0):
            pos[...] = 0
    signal = np.nan_to_num(signal)
    return signal

def use_ref_to_locate_centre(ref_image):
    """Receives a reference image of a 2D gaussian profile and outputs
    the 2D gaussian paramters"""
    #ref = np.array(plt.imread(ref_image),dtype=np.float64)
    #ref = ref[1:]
    #if len(ref.shape) > 2:
    #    ref = ref[:,:,0]
    return fitgaussian2d(ref_image)
    
def use_ref_to_locate_centre_gauss1d(ref_image):
    """Receives a reference image of a 2D gaussian profile and outputs
    the 2D gaussian paramters"""
    refx = np.sum(ref_image,axis=0)
    refy = np.sum(ref_image,axis=1)
    py = fitgaussian1d(None,refy)
    px = fitgaussian1d(None,refx)
    return np.array([py[0]+px[0], py[1], px[1], py[2], px[2], py[3]+px[3]])


def create_array_for_averaging(gauss2D_param,gauss_sigma_frac):
    """Create a 2D matrix with the wanted size to colect the same measurement
    from a set of images"""
    param1 = gauss2D_param
    frac = gauss_sigma_frac
    dx1 = int(param1[4]*frac)
    dy1 = int(param1[3]*frac)
    out = prepare_for_fft_padding(np.zeros((2*dy1,2*dx1)))
    return out

def read_file_to_ndarray(filename):
    """Converts an image from its filename to a ndarray, selecting just last colour channel.
    Removes the first row in the image, due to PTGrey Chameleon acquisition sets some pixels to
    maximum intensity value"""
    signal1 = np.array(plt.imread(filename),dtype=np.float64)
    signal1 = signal1[1:]
    if len(signal1.shape) > 2:
        signal1 = signal1[:,:,0]
    return signal1

def read_file_to_ndarray_keep_channels(filename):
    """Converts an image from its filename to a ndarray, keeping all coulour channels. Removes the first row in the image,
    due to PTGrey Chameleon acquisition sets some pixels to
    maximum intensity value"""
    signal1 = np.array(plt.imread(filename),dtype=np.float64)
    signal1 = signal1[1:]
    return signal1

def do_fft_with_ref(signal_image, gauss2D_param, gauss_sigma_frac):
    """Receives an input image and outputs the fourier transformed image
    and the cropped image used for the FFT. It has some Chameleon tunings"""
    param1 = gauss2D_param
    frac = gauss_sigma_frac
    centre1 = (param1[1],param1[2])
    dx1 = int(np.abs(param1[4]*frac))
    dy1 = int(np.abs(param1[3]*frac))
    
    #signal1 = np.array(plt.imread(signal_image),dtype=np.float64)
    #signal1 = signal1[1:]
    #if len(signal1.shape) > 2:
    #    signal1 = signal1[:,:,0]
    signal1 = signal_image[centre1[0]-dy1:centre1[0]+dy1, centre1[1]-dx1:centre1[1]+dx1]
    #signal1 = signal_image

    signal1 = prepare_for_fft_padding(signal1)
    resft1 = ft.fft2(signal1)
    resft1 = ft.fftshift(resft1)
    resft1 = np.absolute(resft1)

    return resft1, signal1

def imshowfft(subplot,resft,frac,logscale=True,colormap='jet'):
    """Plot using matplotlib imshow the image around zero order pump"""
    y,x = np.shape(resft)
    resft = resft[np.round(y/2 - frac*y/2).astype(int) : np.round(y/2 + frac*y/2).astype(int),
                        np.round(x/2 - frac*x/2).astype(int) : np.round(x/2 + frac*x/2).astype(int)]
    if logscale==True:
        res = subplot.imshow(resft,
               interpolation='none', origin='upper', cmap = colormap,
               norm = LogNorm(vmin=np.amin(resft),
                              vmax=np.amax(resft)))
    else:
        res = subplot.imshow(resft,
               interpolation='none', origin='upper', cmap = colormap,
               vmin=np.amin(resft), vmax=np.amax(resft))
    return res


def do_fft(signal1):
#    signal1 = signal1[1:]
#    signal1 = prepare_for_fft_padding(signal1)
    resft1 = ft.fft2(signal1)
    resft1 = ft.fftshift(resft1)
    resft1 = np.absolute(resft1)

    return resft1


 
def poly2_zero_cross(a, b):
    """"""
    return lambda x: a*x**2 + b*x

def fit_poly2_zero_cross(x,data):
    """Returns (height, centre, sigma, offset)
    the gaussian parameters of a 1D distribution found by a fit"""
    params = np.array([-1,100])
    if (x==None):
        errorfunction = lambda p: poly2_zero_cross(*p)(*np.indices(data.shape))\
                                - data
    else:
        errorfunction = lambda p: poly2_zero_cross(*p)(x)\
                                - data
    p, success = spleastsq(errorfunction, params, full_output=0)
    return p
  
def gets_integration_noise_on_fourier_space(radial_plot,start_pos):
    """Uses a background or alike to integrate the noise in the
    circular integrationa algorithm and returns the parameters of
    the 1D fit."""
    x = np.arange(start_pos,len(radial_plot))
#    xfull = np.arange(len(radial_plot))
    #p = np.polyfit(x, radial_plot[start_pos:],deg=2)
    p = fit_poly2_zero_cross(x,radial_plot[start_pos:])
    p = np.append(p,0)
    
    return np.poly1d(p)
    


def load_files(dname,ext=".bmp"):
    files = []
    for i in os.listdir(dname):
        if i.endswith(ext):
            files = np.append(files,i)
    files.sort()
    print 'Found %d files' %len(files)
    return files
    
def load_files_prefix(dname,prefix,sufix):
    files = []
    for i in os.listdir(dname):
        if i.startswith(prefix) and i.endswith(sufix):
            files = np.append(files,i)
    files.sort()
    print 'Found %d files' %len(files)
    return files


def find_peaks(func,interpolation_points=1000,peak_finding_smoothness=30,
               plot=False, plot_new_fig=True):
    x = np.arange(0,len(func))
    y = func
    f = interp1d(x,y,kind='linear')
    x_2 = np.linspace(0,len(func)-1,interpolation_points)
    y_2 = f(x_2)
    data_obj = Data(x_2,y_2,smoothness=peak_finding_smoothness)
    data_obj.normalize()
    try:
        data_obj.get_peaks(method='slope')
        if plot==True:
            data_obj.plot(new_fig=plot_new_fig)
        return data_obj
    except ValueError:
        return 0

def find_peaks_big_array(func,interpolation_points=1000,peak_finding_smoothness=30,
               plot=False, plot_new_fig=True):
    """Find peaks on 'big' arrays doesn't work when array is normalized...
    So this function doesn't normalize the array before running the peaks
    method."""
    x = np.arange(0,len(func))
    y = func
    f = interp1d(x,y,kind='linear')
    x_2 = np.linspace(0,len(func)-1,interpolation_points)
    y_2 = f(x_2)
    data_obj = Data(x_2,y_2,smoothness=peak_finding_smoothness)
    #data_obj.normalize()
    try:
        data_obj.get_peaks(method='slope')
        if plot==True:
            data_obj.plot(new_fig=plot_new_fig)
        return data_obj
    except ValueError:
        return 0

def detect_peaks(image):
    """
    Takes an image and detect the peaks using the local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask
    detected_peaks = local_max - eroded_background

    return detected_peaks

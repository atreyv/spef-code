#! /usr/bin/python

from PIL import Image
import scipy as scp
import scipy.fftpack as ft
import matplotlib.pyplot as plt
import pylab
import os
import re

lhw = 3.08195610e+02
asym = 3.55334681e-01
#A1 = 0.0

def ut71c_data(name, second):
        """Receives a text filename with output data from ut71c
        multimeter and returns 2 sublists, time in pos zero
        (with values acquired in each second) and value in
        pos one.  """   
        f = open(name,'r')
        line = f.readline()
        value, t = [], []
        n, t_count = 0, 0
        while line != '':
                temp = []
                for i in range(0, len(line)):
                        if line[i] == ':':
                                temp += [int(line[i-2:i])]
                        if line[i:i+2] == ': ':
                                value_temp = float(line[i+2:i+10])
                if len(temp) == 3:
                        t_temp = temp[0]*3600+temp[1]*60+temp[2]
                        if n == 0:
                                tzero = t_temp
                                n=1
                        t_temp -= tzero
                        if t_temp - t_count >= second:
                                t += [t_temp]
                                value += [value_temp]
                                t_count = t_temp
                line = f.readline()
        f.close()
        return t, value

def spinit_raw(name):
        """Receives a text file containning spr profiles
        from spinit-deploy binary (excludes the first) and outputs a matrix
        of floats with a profile per line"""
        f = open(name,'r')
        for n in range(0,1):
                line = f.readline()
        value = []
        j, k = 0, 0
        while line != '':
                if len(line) > 1:
                        value+=[[]]
                for i in range(0, len(line)-3):
                        if (line[i:i+3] == ' 0 ' or line[i:i+3] == ' 1 '):
                                value[j] += [float(line[i:i+3])]
                        if line[i] == '.':
                                while line[i-k] != ' ':
                                        k += 1
                                value[j] += [float(line[i-k:i+7])]
                        if line[i] == ',':
                                while line[i-k] != ' ':
                                        k += 1
                                line_b = line[i-k:i] + '.' + line[i+1:i+7]
                                value[j] += [float(line_b)]
                        k=0
                if len(line) > 1:
                        j += 1
                line = f.readline()
        f.close()
        return scp.array(value)

def spinit_centroid(name):
        """Receives a text file from BIOSPR with centroid values and respective
        time and outputs a vector of pairs"""
        f = open(name,'r')
        line = f.readline()
        value, time = [], []
        i = 0
        for i in range(0, len(line)):
                if (line[0:3] != 'NaN'):
                        value_i = float(line[0:18])
                else:
                        value_i = 0.0
                if line[i:i+1] == '\n':
                        time_i = float(line[i-13:i])/1000.
        while line != '':
                if (line[0:3] != 'NaN'):
                        value += [float(line[0:18]) - value_i]
                else:
                        value = 0.0
                for i in range(0, len(line)):
                        if line[i:i+1] == '\n':
                                time += [float(line[i-13:i])/1000. - time_i]
                line = f.readline()
        f.close()
        return scp.array(time), scp.array(value)

def pcgrate_results(name, param):
        """Reads and extracts the wanted values in xml results from pcgrate grating solver"""
        f = open(name,'r')
        angle, value = [], []
        line = f.readline()
        while line != '':
                if len(line) > 0:
                        for i in range(0,len(line)):
                                if line[i:i+25] == 'Scanning_parameter_value=':
                                        angle += [float(line[i+26:i+34])]
                                if line[i:i+16] == 'Order_number="0"':
                                        for j in range(0, len(line)):
                                                if line[j:j+5] == 'false':
                                                        for k in range(0,len(line)):
                                                                if (param == 'eff'):
                                                                        if line[k:k+14] == 'Efficiency_TM=':
                                                                                value += [float(line[k+15:k+37])]
                                                                elif (param == 'amp'):
                                                                        if line[k:k+13] == 'Amplitude_TM=':
                                                                                value += [float(line[k+14:k+36])]

                line = f.readline()
        f.close()
        return scp.array(angle), scp.array(value)

def ohm_to_celsius(ohm,beta,R_25):
        """Model for NTC thermistor"""
        zero = 273.15
        t_ini = 25.
        celsius = []
        A = R_25 / scp.exp(beta / (t_ini + zero))
        for i in range(0, len(ohm)):
                celsius += [beta / scp.log(ohm[i] / A) - zero]
        return scp.array(celsius)

def integrate_each_row_in_matrix(data, threshold):
        """Integrate each row in matrix and returns a vector of integrations"""
        value = []
        for i in range(0,len(data)):
                temp = 0
                for j in range(0,len(data[i])):
                        temp += data[i][j] - threshold
                value += [temp]
        return scp.array(value)

def integrate_curve(curve, threshold):
        """Integrate curve from a threshold and returns a value"""
        value = 0
        for i in range(0,len(curve)):
                value += curve[i] - threshold
        return value

def derivative(data, order):
        """Receives a vector of pairs and calculates the nth
        order derivative"""
        der = []
        for i in range(1, len(data[0])):
                der += [(data[1][i]-data[1][i-1]) / (data[0][i]-data[0][i-1])]
        return scp.array(data[0][1:len(data[0])]), scp.array(der)

def centroid_quadratic(curve, ratio):
        num, div = 0, 0
        top = scp.amin(curve) + (scp.amax(curve) - scp.amin(curve)) * ratio
        for i in range(0, len(curve)):
                if (curve[i] <= top):
                        num += (top - curve[i]) * (top - curve[i]) * (i+1)
                        div += (top - curve[i]) * (top - curve[i])
        centroid = num/div
        return centroid

def spr_analysis(curve, ratio, slope_size):
        """Calculates centroid, slopes at baseline, full width at baseline
        and contrast"""
        num, div, fwbas = 0, 0, 0
        top = scp.amin(curve) + (scp.amax(curve) - scp.amin(curve)) * ratio
        contrast = scp.amax(curve) - scp.amin(curve)
        if (curve[slope_size/2] > top and curve[len(curve)-slope_size/2] > top):
                for i in range(0,len(curve)):
                        if (curve[i] <= top):
                                num += (top - curve[i]) * (i+1)
                                div += top - curve[i]
                                fwbas += 1
                                if (fwbas == 1):
                                        i_ini=i
                centroid = num/div
        else:
                num, div, i_ini, centroid = 0, 0, 0, 0
#        print fwbas, i_ini, slope_size
        if(i_ini+fwbas+slope_size/2 < len(curve)-1 and fwbas > slope_size and i_ini > slope_size/2):
                x =[]
                for j in range(0, slope_size):
                        x += [j]
                slope_l = scp.polyfit(x, curve[i_ini - slope_size/2 : i_ini + slope_size/2],1)[0]
                slope_r = scp.polyfit(x, curve[i_ini + fwbas - slope_size/2 : i_ini + fwbas + slope_size/2],1)[0]
        else:
                slope_l, slope_r = 0, 0
        minimizer = scp.argmin(curve)
        minimum = scp.amin(curve)
        maximum = scp.amax(curve)
        return centroid, slope_l, slope_r, fwbas, contrast, maximum, minimum, minimizer, div

def frame_column_average(vector_of_pixels, row_size, column_size):
        """Receives a vector of row_size*column_size values and calculates
        the average of row_size coordinates along the vector coordinates;
        returns a vector of averages"""
        line_count = 0
        value = []
        for i in range(0, row_size):
                value += [0]
        for i in range(0, column_size):
                for j in range(0, row_size):
                        value[j] += vector_of_pixels[j+line_count] / float(column_size)
                line_count += row_size
        return scp.array(value)

def average_row_in_matrix(matrix):
        value=[]
        for i in range(0, len(matrix[0])):
                value += [0]
        for i in range(0, len(matrix)):
                for j in range(len(matrix[i])):
                        value[j] += matrix[i][j] / float(len(matrix))
        return scp.array(value)

def curve_ad_ave(profile, row_size, ad_ave):
        """Receives a vector and calculates a moving average in each coordinate with length of ad_ave
        to right and to left; returns a vector with the same size of received one (copied and recoded
        from calc.c from spinit-deploy)"""
        value = []
        if (ad_ave > 1):
                for i in range(0, row_size):
                        value += [0]
                        ini = i - int(ad_ave / 2)
                        end = i + int(ad_ave / 2)
                        if (ini < 0):
                                ini = 0
                        if (end > row_size):
                                end = row_size
                        for j in range(ini, end):
                                value[i] += profile[j] / float(end - ini)
        else:
                value = profile
        return scp.array(value)

def low_pass_fft(curve, low_freqs):
        """Filters the curve by setting to zero the high frequencies"""
        center = len(curve)/2
        a = scp.fft(curve)
        for i in range(0, center - low_freqs):
                a[center+i]=0
                a[center-i]=0
        return scp.array(scp.ifft(a))

def low_pass_rfft(curve, low_freqs):
        """Filters the curve by setting to zero the high frequencies"""
        a = ft.rfft(curve)
        for i in range(2*low_freqs, len(curve)):
                a[i]=0    
        return scp.array(ft.irfft(a))

def gamma_corr(curve):
        """Corrects the intensity axis from companding mode in CMOS (8-bit)
        to linear scale"""
        for i in range(0, len(curve)):
                if(curve[i] < 64):
                        curve[i] = curve[i]/4.
                elif(curve[i] >= 64 and curve[i] < 96):
                        curve[i] = curve[i]/2.-16
                elif(curve[i] >= 96 and curve[i] < 192):
                        curve[i] = curve[i]-64
                elif(curve[i] >= 192 and curve[i] < 256):
                        curve[i] = curve[i]*2.-256
        return scp.array(curve)

def shift_minimums(data1, data2):
        """Receives two graphs, calculates the shift between y-axis minimums in x-axis
        and returns the shift in index and in x-axis units"""
        pos1 = scp.argmin(data1[1])
        pos2 = scp.argmin(data2[1])
        value1 = data1[0][pos1]
        value2 = data2[0][pos2]
        return [pos2-pos1,value2-value1]
                             
def spr_lorentz_total(x,p):
    """Calculates the Lorentz distribution (SPR modulation) for an array of
    pixels. Input are the total parameters in Lorentz distribution"""
    return p[0]-p[1]*(p[2]+p[3]*(x-p[4]))/(p[2]*(1+pow((x-p[4])/p[2],2)))

def residuals_spr_lorentz_total(p,y,x):
    """Calculates the array of residuals from partial (only Imax, Idip and
    x0) Lorentz distribution altered to spr shape"""
    lmax, ldip, lhw, asym, lx0 = p
    err = y - (lmax-ldip*(lhw+asym*(x-lx0))/(lhw*(1+pow((x-lx0)/lhw,2))))
    return err

def spr_lorentz_partial(x,p):
    """Calculates the Lorentz distribution (SPR modulation) at a given x.
    Input are the partial parameters (Imax, Idip and x0)
    in Lorentz distribution"""
    return lmax-p[0]*(lhw+asym*(x-p[1]))/(lhw*(1+pow((x-p[1])/lhw,2)))

def residuals_spr_lorentz_partial(p,y,x):
    """Calculates the array of residuals from total (Imax, Idip, FWHM, asym and
    x0) Lorentz distribution altered to spr shape"""
    ldip, lx0 = p
    err = y - (lmax-ldip*(lhw+asym*(x-lx0))/(lhw*(1+pow((x-lx0)/lhw,2))))
    return err

def gauss(x,p):
    """Calculates the Gauss distribution (SPR modulation) at a given x.
    Input are the p = [FWHM, x0]"""
    return scp.exp(-1*pow((x-p[1])/(2.*p[0]),2))

def inverted_sigmoid(px, p):
    return p[1] * pow((p[3] - px)/(px - p[0]), 1/p[2])

def sigmoid_logistic(x,p):
    return p[0] + (p[3] - p[0]) / (1 + pow(x / p[1], p[2]))

def residuals_sigmoid_logistic(p,y,x):
    A2, x0, power, A1 = p
    err = y - (A2 + (A1-A2)/(1+pow(x/x0,power)))
    return err

def get_files(path,pattern):
    exp_folders = []
    for root, dirs, files in os.walk(path):
        for d in files:
            #search for patterns in file names.
            matchObj = re.search(pattern, d)
            #in case the pattern was found ...
            if matchObj: #and matchObj_2:
                #get the full path
                full_path = os.path.join(root,d) 
                exp_folders.append(full_path)
    return sorted(exp_folders)

# Entry point
if __name__ == '__main__':
        print "Just a library"


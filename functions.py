import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, butter, filtfilt
import pandas as pd
import re

def r_s (theta, n): #n = n2/n1
    """Reflection coefficient for s (TE) polarization.
    
    Args:
        theta: Angle of incidence in radians.
        n: Ratio of the refractive indexes = n2/n1, where light goes from medium 1 -> medium 2.

    Returns:
        Reflection coefficient for s polarization.
    """
    return (np.cos(theta)-np.sqrt(n**2-np.sin(theta)**2+0j)) / (np.cos(theta)+np.sqrt(n**2-np.sin(theta)**2+0j))

def r_p (theta, n):
    """Reflection coefficient for p (TM) polarization
    
    Args:
        theta: Angle of incidence in radians.
        n: Ratio of the refractive indexes = n2/n1, where light goes from medium 1 -> medium 2.

    Returns:
        Reflection coefficient for p polarization.
    """
    return (-n**2*np.cos(theta)+np.sqrt(n**2-np.sin(theta)**2+0j)) / (n**2*np.cos(theta)+np.sqrt(n**2-np.sin(theta)**2+0j))

def t_s (theta, n):
    """Trasmission coefficient for s (TE) polarization
    
    Args:
        theta: Angle of incidence in radians.
        n: Ratio of the refractive indexes = n2/n1, where light goes from medium 1 -> medium 2.

    Returns:
        Transmission coefficient for s polarization.
    """
    return 2*np.cos(theta) / (np.cos(theta)+np.sqrt(n**2-np.sin(theta)**2+0j))

def t_p (theta, n):
    """Transmission coefficient for p (TM) polarization
    
    Args:
        theta: Angle of incidence in radians.
        n: Ratio of the refractive indexes = n2/n1, where light goes from medium 1 -> medium 2.

    Returns:
        Transmission coefficient for p polarization.
    """
    return 2*n*np.cos(theta) / (n**2*np.cos(theta)+np.sqrt(n**2-np.sin(theta)**2+0j))

def T_FP_sem_perda (r1, r2, lamb, n_f, d_f):
    """Trasmittance of Fabry-Perot cavity without loss.
    
    Args:
        r1: Reflection coefficient at the input interface (light going from inside to outside of FP cavity).
        r2: Reflection coefficient at the output interface (light going from inside to outside of FP cavity).
        lamb: Wavelength (same units as d_f).
        n_f: Refractive index of FP cavity.
        d_f: Length of FP cavity (same units as lamb).

    Returns:
        Transmittance of FP cavity without loss.
    """
    phi = 2*np.pi*n_f/lamb * d_f
    return (1-r1**2)*(1-r2**2) / ((1-r1*r2)**2+4*r1*r2*np.sin(phi)**2)

def FP_n_passes (n, n_s, n_f, n_o, lamb, d_f, kappa_f=0.0):
    """Reflectance and transmittance of FP cavity after n passes
    
    Args:
        n: Number of passes through FP cavity.
        n_s: Refractive index of input medium.
        n_f: Refractive index of cavity medium.
        n_o: Refractive index of output medium.
        lamb: Wavelength (same units as d_f).
        d_f: Length of FP cavity (same units as lamb).
        kappa_f: Imaginary part of the complex refractive index of cavity medium.

    Returns:
        A tuple of (Reflectance, Transmittance)
    """
    try:
        Er = r_s(0, n_f/n_s)*np.ones(len(lamb))
    except ValueError:
        Er = r_s(0, n_f/n_s)

    t1 = t_s(0, n_s/n_f)
    t2 = t_s(0, n_o/n_f)
    r1 = r_s(0, n_s/n_f)
    r2 = r_s(0, n_o/n_f)
    
    phi = 2*np.pi*n_f/lamb * d_f
    phi_im = 2*np.pi*kappa_f/lamb * d_f
    E2 = t_s(0, n_f/n_s)*np.exp(-1j*phi-phi_im)
    Et = E2*t2
    
    for i in range(n):
        E1_menos = E2*r2*np.exp(-1j*phi-phi_im)
        Er += E1_menos*t1
        E2 = E1_menos*r1*np.exp(-1j*phi-phi_im)
        Et += E2*t2
        
    return np.absolute(Er)**2, np.absolute(Et)**2*n_o/n_s

def T_FP_com_perda (r1, r2, lamb, n_f, d_f, kappa_f):
    """Trasmittance of Fabry-Perot cavity with loss.
    
    Args:
        r1: Reflection coefficient at the input interface (light going from inside to outside of FP cavity).
        r2: Reflection coefficient at the output interface (light going from inside to outside of FP cavity).
        lamb: Wavelength (same units as d_f).
        n_f: Refractive index of FP cavity.
        d_f: Length of FP cavity (same units as lamb).
        kappa_f: Imaginary part of complex refractive index of cavity medium.

    Returns:
        Transmittance of FP cavity with loss.
    """
    phi = 2*np.pi*n_f/lamb * d_f
    phi_im = 2*np.pi*kappa_f/lamb * d_f
    return (1-r1**2)*(1-r2**2)*np.exp(-2*phi_im) / ((1-r1*r2*np.exp(-2*phi_im))**2+4*r1*r2*np.exp(-2*phi_im)*np.sin(phi)**2)

def R_FP_com_perda (r1, r2, lamb, n_f, d_f, kappa_f):
    """Reflectance of Fabry-Perot cavity with loss.
    
    Args:
        r1: Reflection coefficient at the input interface (light going from inside to outside of FP cavity).
        r2: Reflection coefficient at the output interface (light going from inside to outside of FP cavity).
        lamb: Wavelength (same units as d_f).
        n_f: Refractive index of FP cavity.
        d_f: Length of FP cavity (same units as lamb).
        kappa_f: Imaginary part of complex refractive index of cavity medium.

    Returns:
        Reflectance of FP cavity with loss.
    """
    phi = 2*np.pi*n_f/lamb * d_f
    phi_im = 2*np.pi*kappa_f/lamb * d_f
    x = np.exp(-2*phi_im)
    return (r1**2+r2**2*x**2-2*r1*r2*x*np.cos(2*phi)) / (1+r1**2*r2**2*x**2-2*r1*r2*x*np.cos(2*phi))

def calc_n_1 (n_s, R_M, R_m, n_f_smaller = False):
    """Calculate first approximation of the film's refractive index
    
    Args:
        n_s: substrate's refractive index
        R_M: upper envelope of the reflection spectrum
        R_m: lower envelope of the reflection spectrum
        n_f_smaller: True if n_f < n_s

    Returns:
        Film's refractive index
    """
    sign = (n_f_smaller-0.5)*2
    R_Msqrt, R_msqrt = np.sqrt(R_M), np.sqrt(R_m)
    return n_s*np.sqrt( ((n_s+1)+sign*(n_s-1)*R_msqrt)
        * ((n_s+1)-sign*(n_s-1)*R_Msqrt)
        / ((n_s+1)-sign*(n_s-1)*R_msqrt)
        / ((n_s+1)+sign*(n_s-1)*R_Msqrt) )

def calc_x_M (n_s, n_f, R_M, n_f_smaller = False):
    """Calculate attenuation factor x from R_M"""
    sign = (n_f_smaller-0.5)*2
    R_Msqrt = np.sqrt(R_M)
    return ((n_f+1)/(n_f-1)
            * (sign*(n_s+1)*(n_f-n_s)+(n_s-1)*(n_f+n_s)*R_Msqrt)
            / ((n_s+1)*(n_f+n_s)+sign*(n_s-1)*(n_f-n_s)*R_Msqrt))

def calc_x_m (n_s, n_f, R_m, n_f_smaller = False):
    """Calculate attenuation factor x from R_m"""
    sign = (n_f_smaller-0.5)*2
    R_msqrt = np.sqrt(R_m)
    return ((n_f+1)/(n_f-1)
            * (-sign*(n_s+1)*(n_f-n_s)+(n_s-1)*(n_f+n_s)*R_msqrt)
            / ((n_s+1)*(n_f+n_s)-sign*(n_s-1)*(n_f-n_s)*R_msqrt))

def calc_x_Mm(n_s, n_f, R_M, R_m):
    """Calculate attenuation factor x from R_M and R_m"""
    R_Msqrt = np.sqrt(R_M)
    R_msqrt = np.sqrt(R_m)
    return ((n_f+1)/(n_f-1) * np.sqrt(
              ((n_s+1)**2*(n_f-n_s)**2+(n_s-1)**2*(n_f+n_s)**2*R_Msqrt*R_msqrt)
            / ((n_s+1)**2*(n_f+n_s)**2-(n_s-1)**2*(n_f-n_s)**2*R_Msqrt*R_msqrt)))

def n_Wemple_DiDomenico(lamb, E0, Ed):
    """Wemple-DiDomenico refractive index model.
    
    Args:
        lamb: Wavelength values in nm
        E0: Single oscillator energy in eV
        Ed: Dispersion energy in eV

    Returns:
        Refractive index.
    """
    E = 1239.8419738620933/lamb #E->eV; lamb->nm
    return np.sqrt(1+E0*Ed/(E0**2-E**2))

def alpha_Urbach(lamb, alpha0, Eu):
    """Urbach model of the absorption coefficient.
        
    Args:
        lamb: Wavelength values in nm
        alpha0: absorption coefficient for wavelength -> +inf
        Eu: Urbach energy in eV

    Returns:
        Absorption coefficient.
    """
    E = 1239.8419738620933/lamb #E->eV; lamb->nm
    return alpha0*np.exp(E/Eu)


def mse(y1, y2):
    """Mean squared error."""
    return np.sum((y1-y2)**2)/len(y1)

def rmse(y1, y2):
    """Root mean squared error."""
    return np.sqrt(np.sum((y1-y2)**2)/len(y1))

def linear_func(x, m, b):
    """Linear function with slope 'm' and intercept 'b'."""
    return m*x+b

def calc_values(R, lamb_list, n_s_val = 1.462, interpol:str = 'hermite', n_f_smaller = False, peak_prominence = 0.1):
    """Calculate the refractive index, thickness, order number, maxima, minima, upper and lower envelope of the FP cavity,
      from the reflection spectrum R."""
    picos_R,_ = find_peaks(R, prominence=peak_prominence)
    vales_R,_ = find_peaks(-R, prominence=peak_prominence)
    extr = np.sort(np.append(picos_R, vales_R))

    if interpol == 'hermite':
        R_M_interp = PchipInterpolator(lamb_list[picos_R], R[picos_R])
        R_m_interp = PchipInterpolator(lamb_list[vales_R], R[vales_R])

    else:
        R_M_interp = CubicSpline(lamb_list[picos_R], R[picos_R])
        R_m_interp = CubicSpline(lamb_list[vales_R], R[vales_R])

    if n_f_smaller: #n_f < n_s
        n_f1 = calc_n_1(n_s_val, R_m_interp(lamb_list), R_M_interp(lamb_list))
    else:
        n_f1 = calc_n_1(n_s_val, R_M_interp(lamb_list), R_m_interp(lamb_list))

    m_spacing = np.arange(0, (len(extr))*0.5, 0.5)
    x = 2*n_f1[extr]/lamb_list[extr]
    # p1, cov1 = np.polyfit(x, m_spacing, 1, cov = True)
    p1, cov1 = curve_fit(linear_func, x, m_spacing)
    std1, r2_1 = np.sqrt(np.diag(cov1)), r_squared(m_spacing, linear_func(x, p1[0], p1[1]))

    first_min = ((vales_R[0]<picos_R[0]))%2 #first value is a minimum
    if first_min ^ n_f_smaller: #XOR
        m0_exact = round(p1[1]) #round to nearest integer
    else:
        m0_exact = np.floor(p1[1]) + 0.5 #round to nearest half-integer

    # x_aux = x[:,np.newaxis]
    # slope2, res, _, _ = np.linalg.lstsq(x_aux, m_spacing-m0_exact)
    # sigma2 = res/(len(x)-1)
    # slope_se = np.sqrt(sigma2/np.sum(x**2))
    p2, cov2 = curve_fit(lambda x, m: linear_func(x, m, m0_exact), x, m_spacing)
    std2, r2_2 = np.sqrt(np.diag(cov2)), r_squared(m_spacing, linear_func(x, p2[0], m0_exact))

    n2 = (m0_exact-m_spacing)*lamb_list[extr]/(2*-p2[0])

    # x_M = calc_x_M(n_s_val, n2, R_M_interp(lamb_list[extr]))
    # x_m = calc_x_m(n_s_val, n2, R_m_interp(lamb_list[extr]))
    # x_Mm = calc_x_Mm(n_s_val, n2, R_M_interp(lamb_list[extr]), R_m_interp(lamb_list[extr]))
    #, 'x_M': x_M, 'x_m': x_m, 'x_Mm': x_Mm

    return {'n1':n_f1, 'n2': n2, 'd1': -p1[0], 'd2': -p2[0], 'm0_aprox': p1[1], 'm0_exact': m0_exact,
             'd1_std': std1[0], 'm0_aprox_std': std1[1], 'd2_std': std2[0], 'r2_1': r2_1, 'r2_2': r2_2,
             'max_R': picos_R, 'min_R':vales_R, 'R_M_interp': R_M_interp, 'R_m_interp': R_m_interp}

def r_squared(y_data, y_fit):
    """Coefficient of determination r^2."""
    ss_res = np.sum((y_data - y_fit) ** 2) # residual sum of squares
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2) # total sum of squares

    r2 = 1 - (ss_res / ss_tot) # r-squared
    return r2

def n1_std_estimate(n, R_M, R_m, std_R_M, std_R_m):
    """Estimate the error of the calculation of the refractive index from the upper and lower envelopes."""
    std_n = n*np.sqrt((std_R_M/R_M)**2+(std_R_m/R_m)**2)
    return std_n

def d_std_estimate(d, wav, std_wav, n, std_n):
    """Estimate the error of the calculation of the thickness by propagating the errors associated with the wavelength and refractive index."""
    std_d = d*np.sqrt((std_wav/wav)**2 + (std_n/n)**2)
    return std_d

def d_std_estimate2(d, x, m0_exact):
    """Estimate the error of the calculation of the thickness by the RMSE of the x axis values in relation to the obtained fit."""
    m_spacing = np.arange(0, (len(x))*0.5, 0.5)
    delta_x = (m0_exact-m_spacing)/d-x
    x_std = np.sqrt(np.sum(delta_x**2)/len(x))
    # print(f'x_std = {x_std}')
    d_std = d*x_std/x
    return d_std

def d2_std_estimate(d_fit2_std, m0_std, x):
    """Estimate the error of the calculation of the thickness by considering the error of the intercept from the first fit"""
    m0_contrib = m0_std*(np.sum(x)/np.sum(x**2))
    return np.sqrt(d_fit2_std**2+m0_contrib**2)

def n2_std_estimate(n2, d2, d2_std, wav, wav_std):
    """Estimate of error of the calculation of the 2nd approximation of the refractive index."""
    return  n2*np.sqrt((wav_std/wav)**2 + (d2_std/d2)**2)

def butter_filter(signal, cutoff, order = 5, fs = None, btype = 'low', return_ba = False):
    """Apply a Butteworth filter to the provided signal.
        Args:
            signal: Signal to apply the filter to
            cutoff: Cutoff frequency of the filter
            order (optional): Order of the Butterworth filter
            fs (optional): Sampling frequency of the signal
            btype (optional): Butteworth filter type ('lowpass', 'highpass', 'bandpass', 'bandstop')
            return_ba (optional): Whether or not to return the numerator (b) and denominator (a) polynomials of the Butterworth filter
    
        Returns:
            filtered_signal: Resulting filtered signal
            filtered_signal, b, a: Resulting filtered signal, plus numerator (b) and denominator (a) polynomials of the filter. Returned if return_ba = True
    """
    b, a = butter(order, cutoff, fs=fs, btype=btype, analog=False)

    if return_ba:
        return filtfilt(b, a, signal), b, a
    else:
        return filtfilt(b, a, signal)

def ideal_lowpass(signal, cutoff, fs = 1, return_fft_filt = False):
    """Apply an ideal low-pass filter to the provided signal.
        Args:
            signal: Signal to apply the filter to
            cutoff: Cutoff frequency of the filter
            fs (optional): Sampling frequency of the signal. Default = 1
            return_fft_filt (optional): Whether or not to return the filtered signal's FFT
    
        Returns:
            filtered_signal: Resulting filtered signal
            filtered_signal, filtered_signal_fft: Resulting filtered signal, plus its FFT. Returned if return_fft_filt = True
    """
    s_fft = np.fft.fft(signal)

    freq = np.fft.fftfreq(len(signal), 1/fs)

    s_fft_filter = np.where(np.abs(freq)<=cutoff, s_fft, np.zeros(len(signal)))

    s_filter = np.real(np.fft.ifft(s_fft_filter))

    if return_fft_filt:
        return s_filter, s_fft_filter
    else:
        return s_filter

def trim_df(df: pd.DataFrame, start_val: float, stop_val: float, col_name: str = "Wavelength (nm)") -> pd.DataFrame:
    """Trims the given dataframe, such that start_val < df[col_name] < stop_val."""
    return df.loc[(df[col_name]>start_val) & (df[col_name]<stop_val)].reset_index(drop=True)

def ler_OSA(fich: str) -> pd.DataFrame:
    """Read a CSV file from a Yokogawa AQ6370 OSA.
    
    Args:
        fich: name/path to the CSV file.

    Returns:
        Dataframe with columns
            - `Wavelength (nm)`: wavelength values in nm.
            - `Power lin`: power in linear scale.
            - `Power (dB)`: power in log scale.
    """
    dic, start_row = ler_cabecalho(fich)
#     print(dic)    
    if ("SSCLN" in dic) or ("BASEL" in dic): #linear
        scale = 'linear'
        valor_max = float(dic["SMIN"])+float(dic["SSCLN"])*12
        df = pd.read_csv(fich, names=["Wavelength (nm)", "Power lin"], skiprows = start_row)
        df["Power lin"] = df["Power lin"].where(df["Power lin"]<valor_max, other=valor_max)
        df["Power (dB)"] = 10*np.log10(df["Power lin"].where(df["Power lin"]!=0, other=1e-1))
    
    elif ("SSPS" in dic) or ("SMINP" in dic): #%
        scale = '%'
        df = pd.read_csv(fich, names=["Wavelength (nm)", "Power lin"], skiprows = start_row)
        df["Power lin"] = df["Power lin"]/100
        df["Power (dB)"] = 10*np.log10(df["Power lin"].where(df["Power lin"]!=0, other=1e-1))
    
    elif ("SSKM" in dic) or ("OFSKM" in dic) or ("LENG" in dic): #dB/km
        scale = 'dB/km'
        df = pd.read_csv(fich, names=["Wavelength (nm)", "Power (dB)"], skiprows = start_row)
        df["Power (dB)"] = df["Power (dB)"]*float(dic["LENG"])
        df["Power lin"] = 10**(df["Power (dB)"]/10)

    else: #log
        scale = 'log'
        df = pd.read_csv(fich, names=["Wavelength (nm)", "Power (dB)"], skiprows = start_row)
        df["Power lin"] = 10**(df["Power (dB)"]/10)
        
    # print(scale)
    return df

def is_float(elem):
    """Returns if 'elem' is or can be converted into a float."""
    try:
        float(elem)
        return True
    except (ValueError, TypeError):
        return False
        
def ler_cabecalho(fich: str) -> tuple[dict[str, str], int]:
    """Read the header information from a Yokogawa AQ6370 OSA CSV file.
    
    Args:
        fich: name/path to the CSV file.

    Returns:
        A tuple `(header, start_row)` where
            - `header` is a dictionary containing all the header information.
            - `start_row` is the row index where the measurment data begins.
    """
    dic={}
    start_row = 0
    with open(fich, 'r') as f:
        for n, linha in enumerate(f):
            params = re.split(',', linha.replace('\n', '').replace('"','').replace(' ',''))
            if len(params)>1:
                if is_float(params[0]): #already is data
                    start_row = n
                    break
                else: #is header info
                    dic[params[0]] = params[1]
            
            elif linha == '[TRACE DATA]\n': #data starts on the next line
                start_row = n+1
                break
            
    return dic, start_row

def remove_laser_peak(x: pd.Series, y: pd.Series, start_x: float, stop_x: float):
    """Removes the laser peak from 'y' found between start_x < x < stop_x, by substituting it with a linear regression of the points at start_x and stop_x."""
    ind_min = x[x >= start_x].index[0]
    ind_max = x[x >= stop_x].index[0]

    new_y = y.copy()
    new_y[ind_min:ind_max] = (y.iloc[ind_max]-y.iloc[ind_min])/(x.iloc[ind_max]-x.iloc[ind_min])*(x.iloc[ind_min:ind_max]-x.iloc[ind_min])+y.iloc[ind_min]
    return new_y
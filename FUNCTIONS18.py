#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 15:43:33 2025

@author: greta
"""
# pylint: disable=C

# FUNCTIONS DEF

import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from astropy.io import fits                                                    
from functools import reduce                                                   
import sympy as sp                                                             
from arte.types.guide_source import GuideSource 
from arte.types.aperture import CircularOpticalAperture 
from arte.atmo.von_karman_covariance_calculator import VonKarmanSpatioTemporalCovariance 
from arte.atmo.cn2_profile import Cn2Profile                                                                                     



# Reads the YAML file (where parameters are listed) and returns a dictionary.

def load_parameters(yaml_file):
    
    with open(yaml_file, 'r', encoding='utf-8') as stream:                     
                                                                               
        try: 
            
            parameters = yaml.safe_load(stream)                                
            return parameters
        
        except yaml.YAMLError as exc:                                          
            
            print(exc)
            return None
        

# Function to define the d2 array, whose length depends on the value of T_total.

def funct_d2 (T_total):
    
    d2 = np.zeros(T_total + 1)
    d2[0] = 1
    
    return d2


# Function that returns the numerator and denominator of the transfer function C, 
# expressed as polynomials in Z. The function also returns the numeric definition of Z.
# For the moment, we are considering the control function C as defined below,
# which means that we are using an integral control.

def funct_C (gain, omega_temp_freq_interval, t_0):    
    
    Z_symbolic = sp.symbols('Z')                                                   
    C = (Z_symbolic * gain) / (Z_symbolic - 1)                                     
    
    # Estraggo numeratore e denominatore
    num, den = sp.fraction(C)                                                      
    
    # Ottengo i coefficienti come array NumPy
    n4 = np.array(sp.Poly(num, Z_symbolic).all_coeffs(), dtype=complex)            
    d4 = np.array(sp.Poly(den, Z_symbolic).all_coeffs(), dtype=complex)        
    
    Z_numeric = np.exp(1j * omega_temp_freq_interval * t_0)
    
    return n4, d4, Z_numeric
  
  
# Function to compute the numerator and denominator polynomials of the transfer functions H_r and H_n
# using np.polymul, np.polyadd and polyval (to evaluate them at Z).
# The expressions written below were derived from Equations (4) and (5) (in "Semianalytical error budget 
# for adaptive optics systems with pyramid wavefront sensors", Agapito and Pinna, 2019)
# through appropriate algebraic steps, allowing us to use np.polymul, np.polyadd,
# and np.polyval to construct the numerator and denominator of H_r and H_n.

def transfer_funct(n1, n2, n3, n4, d1, d2, d3, d4, Z, transfer_function_type):    
   
    H_r_coeff_num = reduce(np.polymul, [d1, d2, d3, d4])                       
    H_r_coeff_den = np.polyadd(reduce(np.polymul, [d1, d2, d3, d4]), reduce(np.polymul, [n1, n2, n3, n4]))
    H_n_coeff_num = reduce(np.polymul, [n2, n3, n4, d1])
    H_n_coeff_den = H_r_coeff_den                                              
    
    H_r_num = np.polyval (H_r_coeff_num, Z)                                    
    H_r_den = np.polyval (H_r_coeff_den, Z)
    H_n_num = np.polyval (H_n_coeff_num, Z)
    H_n_den = np.polyval (H_n_coeff_den, Z)
    
    
    if transfer_function_type == "H_r":
        
        H_r = H_r_num/ H_r_den
        return H_r

    if transfer_function_type == "H_n":
        
        H_n = H_n_num/ H_n_den
        return H_n
    
    else: 
        raise ValueError("Transfer_function_type must be one of 'H_r' o 'H_n'")


# Function to obtain the atmosferic PSD for tip and tilt modes

def main250724(rho, theta, aperture_radius, aperture_center, r0, L0, layers_altitude,
               wind_speed, wind_direction, space_freqs, tempor_freqs):

    source = GuideSource((rho, theta), np.inf)                                                                
    aperture = CircularOpticalAperture(aperture_radius, aperture_center)
    cn2_profile = Cn2Profile.from_r0s(r0, L0, layers_altitude, wind_speed, wind_direction)
   
    vk = VonKarmanSpatioTemporalCovariance(
        source1=source, source2=source, aperture1=aperture, aperture2=aperture,
        cn2_profile=cn2_profile, spat_freqs=space_freqs)
    
    tip_psd = vk.getGeneralZernikeCPSD(j=2, k=2, temp_freqs=tempor_freqs)
    tilt_psd = vk.getGeneralZernikeCPSD(j=3, k=3, temp_freqs=tempor_freqs)

    #Alternatively, you can compute a modal matrix of temporal PSDs:   ## il seguente è un modo più "compatto" per trovare la PSD per i vari modi di Zernike
    modes = [2, 3, 4, 5, 6]
    modes_psd = vk.getGeneralZernikeCPSD(j=modes, k=modes, temp_freqs=tempor_freqs)
    #The diagonal elements are the temporal PSDs of tip, tilt, focus, ast1, ast2:
    tip_psd = modes_psd[0, 0, :]         
    tilt_psd = modes_psd[1, 1, :]
    #focus_psd = modes_psd[2, 2, :]
    #ast1_psd = modes_psd[3, 3, :]
    #ast2_psd = modes_psd[4, 4, :]

    PSD_atmo = np.array([tip_psd, tilt_psd])                                   
    return PSD_atmo
  

# Function to calculate the Fitting Error, see Equation (7) (in "Semianalytical error budget 
# for adaptive optics systems with pyramid wavefront sensors", Agapito and Pinna, 2019)

def fitting_variance(fitting_coeff, actuators_number, telescope_diameter, Fried_parameter):
    
    var_fitting = fitting_coeff * actuators_number ** (-0.9) * (telescope_diameter/Fried_parameter) ** (5/3)
    
    return var_fitting


# Function to compute the numerator and denominator coefficients (n4 and d4) of the transfer function C 
# for all actuators, and to return the numeric Z vector from funct_C.

def compute_n4_d4(gain, omega_temp_freq_interval, t_0, actuators_number):
    
    n4_array_example, d4_array_example, _ = funct_C(gain[0], omega_temp_freq_interval, t_0)        
 
    len_n4 =  len(n4_array_example)
    len_d4 =  len(d4_array_example)
 
    n4_array = np.zeros((actuators_number, len_n4), dtype=complex)      
    d4_array = np.zeros((actuators_number, len_d4), dtype=complex)      
       
    for i in range (actuators_number):                                   
         
        n4_array[i, :], d4_array[i, :], Z_num = funct_C (gain[i], omega_temp_freq_interval, t_0)     
         
    return n4_array, d4_array, Z_num
  

# Function to compute the transfer function H.
# The 'transfer_function_type' argument selects between two different types of transfer functions: "H_r" or "H_n".

def compute_H(actuators_number, omega_temp_freq_interval, num1, num2, num3, num4, den1, den2,
              den3, den4, Z, transfer_function_type):
    
    H = np.zeros((actuators_number, len(omega_temp_freq_interval)), dtype=complex)
    
    for i in range(actuators_number):
        
        H[i, :] = transfer_funct(num1, num2, num3, num4[i, :], den1, den2, den3, den4[i, :], Z,
                                transfer_function_type)
            
    return H
    

# Funtion to compute the output PSD by multiplying the squared modulus of the transfert function with the
# input PSD.
# This term appears as the integrand in Equations (8), (10), and (15) in "Semianalytical error budget 
# for adaptive optics systems with pyramid wavefront sensors", Agapito and Pinna (2019).

def func_out(function, function_input):
    
    funtion_output = (np.abs(function)) ** 2 * (function_input)   
    
    return funtion_output
    

# Function to compute the numerical integral of the output PSD over the interval of 
# the considered frequencies using Simpson's rule.

def integrate_function(integrand_function, integr_interval): 
    
    result = integrate.simpson(integrand_function, integr_interval)
    
    return result
    

# Function that allows us to load the windshake PSD and their corresponding frequencies

def load_PSD_windshake(file_path_wind): 
    
    with fits.open(file_path_wind) as hdul: 
        
        try: 
            
            data = hdul[0].data                                                # pylint: disable=E1101 
            frequencies = data[0, :]                                           # first row: frequencies 
            PSD_windshake = data[1:, :]                                        # second and third rows: PSD tip and tilt

            return frequencies, PSD_windshake

        except Exception as exc: 
            
            print(exc)
            return None, None
    

# Function to resize the vibration PSD to match the size of the atmospheric turbulence PSD 
# by filling with zeros, if needed.

def resize_psd_like(PSD_atmo_turb, PSD_vibration):
    
    PSD_vib_1= np.zeros_like(PSD_atmo_turb)  
    m = PSD_vibration.shape[0]  
    PSD_vib_1[:m, :] = PSD_vibration
    
    return PSD_vib_1


# Computes the output PSD by applying the corresponding transfer function to the input PSD 
# (using the function "func_out"). Than, the function integrates each modal output PSD over 
# the given frequency range, as described in Equation (8), (10) and (15) in "Semianalytical error budget 
# for adaptive optics systems with pyramid wavefront sensors", Agapito and Pinna (2019).

def compute_output_PSD_and_integrate(actuators_number, transf_funct, PSD_input, omega_temp_freq_interval):
    
    variance_ = 0 
    
    PSD_out = np.zeros_like(PSD_input)
    
    for i in range (actuators_number):                                  
         
        PSD_out[i, :] = func_out(transf_funct[i, :] , PSD_input[i, :])
        integral = integrate_function(PSD_out[i, :], omega_temp_freq_interval)
        variance_ += integral
         
    return variance_, PSD_out


# Computes the temporal variance by resizing the vibration PSD, summing atmospheric turbulence PSD 
# and vibration PSD, applying the transfer function, and integrating over the frequency interval.
# See Equation (8) (in "Semianalytical error budget for adaptive optics systems with pyramid wavefront sensors", 
# Agapito and Pinna, 2019)

def temporal_variance (PSD_atmo_turb, PSD_vibration, transf_funct, actuators_number, omega_temp_freq_interval): 
    
    PSD_vib = resize_psd_like(PSD_atmo_turb, PSD_vibration)
    
    PSD_input = PSD_atmo_turb + PSD_vib

    variance_temp, PSD_output = compute_output_PSD_and_integrate(actuators_number, transf_funct, PSD_input, omega_temp_freq_interval)
   
    return variance_temp, PSD_output, PSD_input 


# Computes the aliasing variance by applying the transfer function to the aliasing PSD 
# and integrating the output over the specified frequency interval.
# See Equation (15) (in "Semianalytical error budget for adaptive optics systems
# with pyramid wavefront sensors", Agapito and Pinna, 2019).  

def aliasing_variance (PSD_alias, transf_funct, actuators_number, omega_temp_freq_interval):
    
    PSD_input = PSD_alias
    
    variance_alias, PSD_output = compute_output_PSD_and_integrate(actuators_number, transf_funct, 
                                                                  PSD_input, omega_temp_freq_interval)
    
    return variance_alias, PSD_output, PSD_input 
    

# Computes the pixel variance, as described in Equation (14) in "Semianalytical error budget 
# for adaptive optics systems with pyramid wavefront sensors", Agapito and Pinna (2019). 
# Than computes the slope noise variance, using the pixel variance and weighting 
# pixel noise by their positions, as described in Equation (12) in "Semianalytical error budget 
# for adaptive optics systems with pyramid wavefront sensors", Agapito and Pinna (2019).

def compute_slope_noise_variance(F_excess, n_phot_pix, pixel_pos, sky_bkg, dark_curr, read_out_noise):
    
    pixel_pos = np.array(pixel_pos)                                            
    pixel_variance = F_excess ** 2 * (n_phot_pix + sky_bkg + dark_curr) + read_out_noise
    pix_intensity = n_phot_pix
    slope_variance = np.sum((pixel_pos ** 2) * pixel_variance) / (4 * (pix_intensity)) ** 2    
    
    return slope_variance
 

# Function to extract the noise propagation coefficients from the reconstruction matrix.

def extract_propagation_coefficients(file_path_matrix_R): 
    
    with fits.open(file_path_matrix_R) as hdul: 
        
        try: 
            
            R = hdul[1].data                       # pylint: disable=E1101     
            return np.diag(R @ R.T)

        except Exception as exc:
            
            print(exc) 
            return None

# Function which returns the noise PSD (PSD_w), assuming that the noise w has a flat temporal
# PSD over the entire frequency range, as stated in "Semianalytical error budget 
# for adaptive optics systems with pyramid wavefront sensors", Agapito and Pinna (2019).

def compute_noise_PSD(p_coefficient, omega_temp_freq_interval, actuators_number, sigma2_w):
    
    PSD_w = np.zeros((len(p_coefficient), len(omega_temp_freq_interval)))   
  
    omega_interval_min = np.min(omega_temp_freq_interval)                  
    omega_interval_max = np.max(omega_temp_freq_interval)
   
    for i in range (actuators_number):  

        PSD_w[i, :] = sigma2_w[i] /(omega_interval_max -  omega_interval_min) 

    return PSD_w 


# Calculates the slope noise variance, extracts the propagation coefficients from the FITS
# file, and computes the measurement noise variance for each mode. 
# Then, it calculates the corresponding measurement noise PSD, to obtain the total measurement 
# variance, as described in Equation (10) (in "Semianalytical error budget 
# for adaptive optics systems with pyramid wavefront sensors", Agapito and Pinna (2019). 

def measure_variance (F_excess, n_phot_pix, pixel_pos, sky_bkg, dark_curr, read_out_noise, 
                      file_path_matrix_R, omega_temp_freq_interval, transf_funct, actuators_number):
    
    variance_meas = 0
    
    slope_noise_variance = compute_slope_noise_variance(F_excess, n_phot_pix, pixel_pos, sky_bkg, dark_curr, read_out_noise)
   
    p_coefficient = extract_propagation_coefficients(file_path_matrix_R)
    
    if p_coefficient is None:                                                 
        
        raise RuntimeError("Propagation coefficients not loaded") 
   
    print("Propagation coefficients loaded successfully.\n")
    
    sigma2_w = p_coefficient * slope_noise_variance
    
    PSD_input =  compute_noise_PSD (p_coefficient, omega_temp_freq_interval, actuators_number, sigma2_w)
    
    variance_meas, PSD_output = compute_output_PSD_and_integrate(actuators_number, transf_funct, PSD_input, omega_temp_freq_interval)
    
    return variance_meas, PSD_output, PSD_input 
    

# Function to compute the transfer function H; it internally computes n4 and d4 
# (numerator and denominator of function C) and then builds H using these polynomials.

def build_transfer_function(gain, omega_temp_freq_interval, t_0, actuators_number, num1, num2, num3, den1, 
                            den2, den3, transfer_function_type):   
    
    num4, den4, Z = compute_n4_d4(gain, omega_temp_freq_interval, t_0, actuators_number)

    transfer_function = compute_H (actuators_number, omega_temp_freq_interval, num1, num2, num3, num4, den1, 
                                den2, den3, den4, Z, transfer_function_type)
       
    return transfer_function


# Computes the variance of a given type ('fitting', 'temp', 'alias', or 'meas') for the system.

def variance(omega_temp_freq_interval, t_0, gain, 
             num1, num2, num3, den1, den2, den3, variance_type, actuators_number, telescope_diameter, 
             Fried_parameter, F_excess, sky_bkg, dark_curr, read_out_noise, n_phot_pix, pixel_pos, fitting_coeff, 
             transfer_function_type, PSD_tur=None, PSD_vib=None, PSD_alias=None, file_path_matrix_R=None):
    
   
    valid_types = ("fitting", "temp", "alias", "meas")
    
    if variance_type not in valid_types: 
        raise ValueError("Variance_type must be one of: 'fitting', 'temp', 'alias', or 'meas'")
       
       
    if variance_type == "fitting" and PSD_tur is None and PSD_vib is None and PSD_alias is None:
            
        sigma2_fit = fitting_variance(fitting_coeff, actuators_number, telescope_diameter, Fried_parameter) 
       
        print("Fitting:", sigma2_fit)   
        return sigma2_fit
   
    
    if variance_type == "temp" and PSD_tur is not None and PSD_vib is not None:
    
        H = build_transfer_function(gain, omega_temp_freq_interval, t_0, actuators_number, num1, num2, num3, den1, 
                                   den2, den3, transfer_function_type)
    
        sigma2_temp, PSD_out, PSD_in = temporal_variance (PSD_tur, PSD_vib, H, actuators_number, omega_temp_freq_interval)
         
        print("Temporal:", sigma2_temp)       
        return sigma2_temp, PSD_out, PSD_in, H
   
    
    if variance_type == "alias" and PSD_alias is not None:
        
        H = build_transfer_function(gain, omega_temp_freq_interval, t_0, actuators_number, num1, num2, num3, den1, 
                                   den2, den3,  transfer_function_type)
 
        sigma2_alias, PSD_out, PSD_in = aliasing_variance (PSD_alias, H, actuators_number, omega_temp_freq_interval)
       
        print("Aliasing:", sigma2_alias)       
        return sigma2_alias, PSD_out, PSD_in, H
   
    
   
    if variance_type == "meas" and PSD_tur is None and PSD_vib is None and PSD_alias is None and file_path_matrix_R is not None:
       
        H = build_transfer_function(gain, omega_temp_freq_interval, t_0, actuators_number, num1, num2, num3, den1, 
                                  den2, den3, transfer_function_type)  
      
        sigma2_meas, PSD_out, PSD_in  = measure_variance (F_excess, n_phot_pix, pixel_pos, sky_bkg, dark_curr, 
                                                        read_out_noise, file_path_matrix_R, omega_temp_freq_interval, 
                                                        H, actuators_number)
      
      
        print("Measure:", sigma2_meas)       
        return sigma2_meas, PSD_out, PSD_in, H
      
      

# Function to interpolate a 1D vector to a new set of points, setting values outside the original range to 0.

def interpolate_vector(x_interpolation, x_original, vector_original):
    
    vector_interpolated = np.interp (x_interpolation, x_original, vector_original, left = 0, right = 0) 
    
    return vector_interpolated


# Function to interpolate and normalize PSD to a new frequency interval.

def interpolate_and_normalize_psd(freqs_interpolation, freqs_original, PSD_original, actuators_number):
    
    PSD_interpolated = np.zeros_like(PSD_original)
    PSD_interpolated_normalized = np.zeros_like(PSD_original)
    sigma2 = np.zeros(actuators_number)
    sigma2_interp = np.zeros(actuators_number)
    
    Omega_freqs_interpolation = 2 * np.pi * freqs_interpolation  
    Omega_freqs_original = 2 * np.pi * freqs_original
    
    for i in range (actuators_number):
        
        PSD_interpolated[i, :]=interpolate_vector(freqs_interpolation, freqs_original, PSD_original[i, :])

        sigma2 [i] = integrate.simpson(PSD_original[i, :], Omega_freqs_original)
        sigma2_interp[i] = integrate.simpson(PSD_interpolated[i, :], Omega_freqs_interpolation)
   
        PSD_interpolated_normalized [i, :]= (PSD_interpolated[i, :]) * (sigma2[i])/(sigma2_interp[i])

    return PSD_interpolated_normalized
         

# Function to compute the total variance by summing fitting variance, temporal variance, and meas variance contributions.


def total_variance(fit_err, temp_err, meas_err):
    var_tot = fit_err + temp_err + meas_err 
    print ("La varianza totale è:", var_tot)
    return var_tot 


# Defines a function that allows, when needed, to plot PSD_in, PSD_out, and the transfer function for the variances (temp, alias, meas)

def plot(f, actuators_number, H_r_t, H_n_m, PSD_in_t, PSD_out_t, PSD_in_m, PSD_out_m):
    
    do_plot = input("Do you want to plot PSD_in, PSD_out, and the transfer function? (y/n):")

    if do_plot == "y":
      
        PSD_in = [PSD_in_t, PSD_in_m]                  
        PSD_out = [PSD_out_t, PSD_out_m]               
        H = [H_r_t, H_n_m]
       
        labels_PSD = ["temp", "meas"]
        labels_H = ["r", "n"]
       
        for i in range (actuators_number):                                                            
          
            plt.loglog(f, PSD_in[i][0, :], label=f"PSD_in_{labels_PSD[i]} (mode 0)")       
            plt.loglog(f, PSD_out[i][0, :], label=f"PSD_out_{labels_PSD[i]} (mode 0)")
            plt.xlabel("Frequecy [Hz]")
            plt.ylabel("PSD")
            plt.title(f"PSD {labels_PSD[i]} (modo 0)")
            plt.legend()
            plt.grid()
            plt.show()
           
            plt.loglog(f, np.abs(H[i][0, :])**2)     
            plt.xlabel("Frequency [Hz]")
            plt.ylabel(f"|H_{labels_H[i]}|^2")
            plt.title(f"Transfert function H_{labels_H[i]} (mode 0)")
            plt.grid()
            plt.show()
       

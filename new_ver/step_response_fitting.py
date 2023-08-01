from scipy.optimize import least_squares
import numpy as np
from scipy.special import erfc

def step_response_fitting(waveform, param):
    type, _, coef_B, coef_tau, time_offset, amp_offset = param
    if type in ['lowpass', 'highpass']:
        fitresult = IIR_fitting_RLC(waveform, coef_B, coef_tau, time_offset, amp_offset)
    elif type == 'line':
        fitresult = IIR_fitting_Line(waveform, coef_B, time_offset, amp_offset)
    elif type == 'skin':
        fitresult = IIR_fitting_SkinE(waveform, coef_B, time_offset, amp_offset)
    return fitresult

def IIR_fitting_Line(waveform, coef_B, time_offset, amp_offset):
    def fit_func(params, x):
        B = params
        return amp_offset + B * (np.array(x) - time_offset)
    # Set up the options for the optimization
    options = {
        'method': 'trf',   # Trust-Region
        'loss': 'soft_l1' # Like LAR Robust
    }

    # Define the residual function
    def residual(params, x, y):
        return fit_func(params, x) - y

    # Perform the fitting using least squares
    x_data = waveform[0]
    y_data = waveform[1]
    initial_params = [coef_B]
    result = least_squares(residual, initial_params, args=(x_data, y_data), **options)
    fitresult = result.x

    return fitresult

def IIR_fitting_RLC(waveform, coef_B, coef_tau, time_offset, amp_offset):
    # Define the fitting function
    def fit_func(params, x):
        B, tau = params
        # A, B, C = params
        return amp_offset + B * np.exp(-(np.array(x) - time_offset) / tau)
    
    # Set up the options for the optimization
    options = {
        'method': 'trf',   # Trust-Region
        'loss': 'soft_l1' # Like LAR Robust
    }

    # Define the residual function
    def residual(params, x, y):
        return fit_func(params, x) - y
    
    # Perform the fitting using least squares
    x_data = waveform[0]
    y_data = waveform[1]
    initial_params = [coef_B, coef_tau]
    result = least_squares(residual, initial_params, args=(x_data, y_data), **options)
    fitresult = result.x

    return fitresult

def IIR_fitting_SkinE(waveform, coef_B, time_offset, amp_offset):
    def fit_func(params, x):
        alpha = params
        return amp_offset - erfc(coef_B/(np.sqrt(np.array(x) - time_offset)))
    
    # Set up the options for the optimization
    options = {
        'method': 'trf',   # Trust-Region
        'loss': 'soft_l1' # Like LAR Robust
    }

    # Define the residual function
    def residual(params, x, y):
        return fit_func(params, x) - y

    # Perform the fitting using least squares
    x_data = waveform[0]
    y_data = waveform[1]
    initial_params = [coef_B]
    result = least_squares(residual, initial_params, args=(x_data, y_data), **options)
    fitresult = result.x

    return fitresult
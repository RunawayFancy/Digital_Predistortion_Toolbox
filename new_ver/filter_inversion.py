import step_response_fitting as ft
from fitting_model import *
import predistortion_rec as pdr
from NLab import *

import csv
import matplotlib.pyplot as plt
import numpy as np

def predistor(waveform_full, waveform_segment, param):
    _filter = get_inverse_filter(waveform_segment, param)
    return _filter.filering(waveform_full)

def get_inverse_filter(waveform, param):
    IIR = 'IIR.CSV'
    FIR = None
    # Fit the waveform
    fitresult = ft.step_response_fitting(waveform, param)
    plot_fitted_waveform(waveform, fitresult, param)
    print(fitresult)
    # Get corresponding filter coefficients
    IIR_coef = get_iir_coef(fitresult, param)
    write_in_csv([IIR_coef],IIR)

    return pdr.Filter(param[1], IIR, FIR)


def get_waveform_segment(start_time, end_time, waveform):
    start_point = np.where(np.array(waveform[0]) > start_time)[0][0]
    end_point = np.where(np.array(waveform[0]) > end_time)[0][0] 
    
    time_segment = waveform[0][start_point:end_point]
    amp_segment = waveform[1][start_point:end_point]

    return [time_segment, amp_segment]

def get_iir_coef(fitresult, param):
    type, fs, _, _, _, amp_offset = param
    Ts = 1/fs
    if type in ['highpass', 'lowpass']:
        A = amp_offset
        B, tau = fitresult[0], fitresult[1]
        # calculate the coefficients
        lambda_val = 2 * A * tau + 2 * B * tau + A * Ts
        a1 = (2 * A * tau + 2 * B * tau - A * Ts) / lambda_val
        b0 = (2 * tau + Ts) / lambda_val
        b1 = (-2 * tau + Ts) / lambda_val
    elif type == 'line':
        B = fitresult[0]
        lambda_val = B*Ts +2
        a1 = - (B*Ts - 2)/lambda_val
        b0 = 2/lambda_val
        b1 = -2/lambda_val
    elif type == 'skin':
        pass
    
    a0 = 1
    return [a0, b0, a1, b1]

def get_fitted_amp(waveform, fitresult, param):
    type = param[0]
    if type in ['lowpass', 'highpass']:
        amp_fitted = lowhigh_pass_reponse(waveform[0], fitresult, param)
    elif type == 'line':
        amp_fitted = Line_response(waveform[0], fitresult, param)
    elif type == 'skin':
        amp_fitted = skin_response(waveform[0], fitresult, param)
    return amp_fitted

def plot_fitted_waveform(waveform_segment, fitresult, param):
    plt.figure()
    plt.plot(waveform_segment[0], get_fitted_amp(waveform_segment, fitresult, param), label='Fitting Curve')
    
    # Plot the selected fitting data
    plt.scatter(waveform_segment[0], waveform_segment[1], c='red', label='Fitting Data')
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Fitting Curve and Data')
    plt.legend()
    plt.grid(True)
    
    plt.show()

def write_in_csv(my_list, file_name):
    # Create a CSV file and write the coefficients
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the coefficients row by row
        for i in range(int(len(my_list[0])/2)):
            row = []
            for j in range(len(my_list)):
                row.append(my_list[j][2*i])
                row.append(my_list[j][2*i+1])
            writer.writerow(row)
    return 0
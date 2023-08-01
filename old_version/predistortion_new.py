import numpy as np
import csv
import math as m
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.special import erfc

class Filter:
    """
    The filter is used to predistort the sampled signal
    to correct distortions caused by electronic elements
    such as transmission line, low-pass filter, Bias-tee, etc. 
    
    ----------
    Filter design method

    For IIR filter
    Fitting the step response and get the corresponding coefficients;
        RLC: f(x) = A + B * exp(-t/tau);
        Skin effect: f(x) = A + B * erfc(?)
    A, B, and tau;
    Determine a sampling period Ts;
    Construct a inverse IIR filter by bilinear transform;
    
    For FIR filter
    Construct a inverse FIR filter by invert the transfer function matrix;

    ----------
    Parameters
    
    IIR: str 'IIR.csv'
        multiple column csv file [X1, Y1, X2, Y2, ... Xn, Yn]
        for n-IIR filter
        each set of [x, Y] belongs to one filter
        if =None, no IIR filtering
    fs: float
        Sampling frequency, Ts = 1/fs: sampling period
    FIR: str 'FIR.csv'
        if =None, no FIR filtering
    """

    # Ignore the skin effect
    def __init__(self, fs, IIR, FIR) -> None:
        self.fs = fs
        # Readout IIR coefficients
        if IIR is not None:
            IIR_coefficient = []
            with open(IIR) as csvfile:
                spamreader = csv.reader(csvfile)
                j=0
                for row in spamreader:
                    if j==0:
                        num_filter = int(len(row)/2)
                    for i in range(num_filter):
                        if j == 0:
                            IIR_coefficient.append([])
                        IIR_coefficient[i].append(float(row[2 * i]))
                        IIR_coefficient[i].append(float(row[2 * i + 1]))
                    j+=1
            self.IIR = IIR_coefficient
            self.num_IIR_filter = num_filter
        else:
            self.IIR = None
        # Readout FIR matrix
        if FIR is not None:
            FIR_matrix = []
            with open(FIR) as csvfile:
                spamreader = csv.reader(csvfile)
                for row in spamreader:
                    column_len = len(row)
                    FIR_matrix.append([])
                    for i in range(column_len):
                        FIR_matrix[i].append(float(row[i]))
            self.FIR = FIR_matrix
        else:
            self.FIR = None

    def filering(self, data):
        if self.IIR is not None:
            for i in range(self.num_IIR_filter):
                y = IIR_filtering(self.IIR[i][1], self.IIR[i][3], self.IIR[i][2], data)
                data[1] = y
        # if self.FIR is not None:
        return data

def IIR_filtering(b0, b1, a1, data):
    y_filtered = []
    for j in range(len(data[0])):
        if j == 0:
            y_filtered.append(
                b0 * data[1][j]
            )
        else:
            y_filtered.append(
                b0 * data[1][j] + b1 * data[1][j-1] + a1 * y_filtered[j-1]
            )
    return y_filtered

def write_2d_list_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(data)

def filter_type_exam(type, index_iteration):
    # Define the filter cases using a dictionary
    mean_tau_l = 5 * 10**(-10)
    mean_tau_h = 6 * 10**(-3)#/index_iteration**3
    # mean_tau_h = 6 * 10**(-5)#/index_iteration**3
    filter_cases_RLC = {
        1: {'B': -0.2, 'tau': mean_tau_l},  # low-pass
        2: {'B': 1, 'tau': mean_tau_h},  # high-pass
        3: {'B': 1, 'tau': None}, # Line
        4: {'B': 0, 'tau': None}
    }
    # Select desired filter type
    if (type) in filter_cases_RLC:
        case = filter_cases_RLC[type]
        B = case['B']
        tau = case['tau']
    else:
        print("Invalid filter type selected.")
    return B, tau

def get_data_segment(waveform, type):
    if type  == 1:
    # Read out the starting and ending index of the first wavelet
        start_point = np.where(np.array(waveform[0]) > 4.004*10**(-5))[0][0]
        end_point = np.where(np.array(waveform[0][start_point:]) > 4.05*10**(-5))[0][0] -1 + start_point
    elif type == 2: 
        start_point = np.where(np.array(waveform[0]) > 4.003*10**(1))[0][0]
        end_point = np.where(np.array(waveform[0][start_point:]) > 4.2*10**(1) )[0][0] - 1 + start_point 
    # elif type == 2: 
    #     start_point = np.where(np.array(waveform[0]) > 6.03*10**(-5))[0][0]
    #     end_point = np.where(np.array(waveform[0][start_point:]) > 6.21*10**(-5) )[0][0] - 1 + start_point 

    elif type == 3:
        start_point = np.where(np.array(waveform[0]) > 4.2*10**(1))[0][0] +100 # >1
        end_point = np.where(np.array(waveform[0][start_point:]) >8*10**(1) )[0][0] - 1 + start_point
    else: # For skin effect
        start_point = np.where(np.array(waveform[0]) > 4.004*10**(1))[0][0]
        end_point = np.where(np.array(waveform[0][start_point:]) > 4.02*10**(1) )[0][0] - 1 + start_point 
    return start_point, end_point, end_point - start_point + 1

def step_response_fitting(type, t_fitting_data, y_fitting_data, start_B, start_tau, offset):
    if type in [1,2]:
        fitresult = IIR_fitting_RLC(t_fitting_data, y_fitting_data, start_B, start_tau, offset, type)
    elif type ==3:
        fitresult = IIR_fitting_Line(t_fitting_data, y_fitting_data, start_B, offset)
    else:
        fitresult = IIR_fitting_SkinE(t_fitting_data, y_fitting_data, start_B, offset)
    return fitresult

def get_partial_data_segment(waveform, i, num_point, end_point, num_iteration,signal_segment):
    
    signal_segment = int(signal_segment + num_point * 2**(-i))

    if i == num_iteration:
        signal_segment = num_point

    start_point_p = end_point - signal_segment + 1
    t_fitting_data = waveform[0][start_point_p:end_point]
    y_fitting_data = waveform[1][start_point_p:end_point]

    return t_fitting_data, y_fitting_data

def step_response_Line(t,B,offset):
    return 1.0093 + B*(np.array(t)-offset)

def step_response_lowp(t, fitresult, offset):
    return 1 + fitresult[0] * np.exp(-(np.array(t) - offset)/fitresult[1])

# def step_response_highp(t, fitresult, offset):
#     return 0.9814 + fitresult[0] * np.exp(-(np.array(t) - offset)/fitresult[1])
def step_response_highp(t, fitresult, offset):
    return 1 + fitresult[0] * np.exp(-(np.array(t) - offset)/fitresult[1])

def step_response_skin(t, fitresult, offset):
    return erfc(fitresult[0]/(np.sqrt(np.array(t) - offset)))

def plot_fit_and_data(t_data, y_data, fitresult, offset, type):
    # Plot the fitting curve
    t_fit = t_data
    if type ==3:
        y_fit = step_response_Line(t_fit, fitresult[0], offset)
    elif type == 2:
        y_fit = step_response_highp(t_fit, fitresult, offset)
    elif type ==1 :
        y_fit = step_response_lowp(t_fit, fitresult, offset)
    else:
        y_fit = step_response_skin(t = t_fit, fitresult= fitresult, offset= offset)
    plt.figure()
    plt.plot(t_fit, y_fit, label='Fitting Curve')
    
    # Plot the selected fitting data
    plt.scatter(t_data, y_data, c='red', label='Fitting Data')
    
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

def calculate_coefficients(fitresult, fs, type):
    # Calculating the coefficients of the first-order inverse IIR filter's transfer function
    Ts = 1 / fs  # Sampling period of the designed filter, unit in seconds
    if type in [1,2]:
        A = int(type) % 2
        if type ==2 :
            A = 0.99
        B = fitresult[0]
        tau = fitresult[1]
        lambda_val = 2 * A * tau + 2 * B * tau + A * Ts
        a1 = (2 * A * tau + 2 * B * tau - A * Ts) / lambda_val
        b0 = (2 * tau + Ts) / lambda_val
        b1 = (-2 * tau + Ts) / lambda_val
    elif type ==3:
        B = fitresult[0]
        lambda_val = B*Ts +2
        a1 = - (B*Ts - 2)/lambda_val
        b0 = 2/lambda_val
        b1 = -2/lambda_val
    # print([a1, b0, b1])
    elif type==4:
        alpha = fitresult[0]
        lambda_val = 1 - alpha * np.sqrt(2/Ts) + 4*alpha**2/Ts
        a1 = 1
        b0 =  (1/2 + 4 * alpha**2/Ts) 
        b1 =  (1/2 - 4 * alpha**2/Ts) 

    return a1, b0, b1

def IIR_fitting_Line(t_data, y_data, star_B, offset):
    def fit_func(params, x):
        B = params
        return 1.0093 + B * (np.array(x) - offset)
    # Set up the options for the optimization
    options = {
        'method': 'trf',   # Trust-Region
        'loss': 'soft_l1' # Like LAR Robust
    }

    # Define the residual function
    def residual(params, x, y):
        return fit_func(params, x) - y

    # Perform the fitting using least squares
    x_data = t_data
    y_data = y_data
    initial_params = [star_B]
    result = least_squares(residual, initial_params, args=(x_data, y_data), **options)
    fitresult = result.x

    return fitresult

def IIR_fitting_RLC(t_data, y_data, star_B, star_tau, offset ,type):
    # Define the fitting function
    def fit_func_highp(params, x):
        B, C = params
        # A, B, C = params
        return 1 + B * np.exp(-(np.array(x) - offset) / C)
    #  def fit_func_highp(params, x):
    #     B, C = params
    #     # A, B, C = params
    #     return 0.9814 + B * np.exp(-(np.array(x) - offset) / C)
    
    def fit_func_lowp(params, x):
        B, C = params
        return 1 + B * np.exp(-(np.array(x) - offset) / C)

    # Set up the options for the optimization
    options = {
        'method': 'trf',   # Trust-Region
        'loss': 'soft_l1' # Like LAR Robust
    }

    # Define the residual function
    def residual_highp(params, x, y):
        return fit_func_highp(params, x) - y
    def residual_lowp(params, x, y):
        return fit_func_lowp(params, x) - y

    # Perform the fitting using least squares
    x_data = t_data
    y_data = y_data
    if type == 2:
        initial_params = [ star_B, star_tau] 
        result = least_squares(residual_highp, initial_params, args=(x_data, y_data), **options)
    else:
        initial_params = [star_B, star_tau]
        result = least_squares(residual_lowp, initial_params, args=(x_data, y_data), **options)
    fitresult = result.x

    return fitresult

def IIR_fitting_SkinE(t_data, y_data, alpha, offset):
    def fit_func(params, x):
        alpha = params
        return erfc(alpha/(np.sqrt(np.array(x) - offset)))
    
    # Set up the options for the optimization
    options = {
        'method': 'trf',   # Trust-Region
        'loss': 'soft_l1' # Like LAR Robust
    }

    # Define the residual function
    def residual(params, x, y):
        return fit_func(params, x) - y

    # Perform the fitting using least squares
    x_data = t_data
    y_data = y_data
    initial_params = [alpha]
    result = least_squares(residual, initial_params, args=(x_data, y_data), **options)
    fitresult = result.x

    return fitresult


def get_inverse_filter(waveform, filter_sequence, fs):
    # filter_sequence: list
    # 1: lowpass; 2: highpass
    IIR = 'IIR.csv'
    # FIR = 'FIR.csv'
    FIR = None
    # Use one IIR filter first
    IIR_coeffi = []
    for type in filter_sequence:
        """
        Call the IIR_fitting function and store the returned values
        1: lowpass, 2: highpass, 3: line, 4: skin effect
        """
        start_point, end_point, num_point = get_data_segment(waveform, type)

        if type in [3,4]:
            num_iteration = 1  # Number of iterations you want to perform
        else:
            num_iteration = 1
        signal_segment = 0

        for i in range(1, num_iteration + 1):

            # Get partial fitting data segment
            t_fitting_data, y_fitting_data = get_partial_data_segment(waveform, i, num_point, end_point, num_iteration, signal_segment)

            # Define the offset of the pulse
            if type in [1,3]:
                offset = 4.2*10**(1)
                # offset = waveform[0][start_point]
            else:
                offset = 4.003*10**(1)
            # Get the initial fitting coefficient only for the first iteration
            if i ==1:
                start_B, start_tau = filter_type_exam(type, i)
            # Fitting 
            if i == 4: # The fitting coefficients of skin effects
                start_alpha = 1
            fitresult = step_response_fitting(type, t_fitting_data, y_fitting_data, start_B, start_tau, offset)

            # # Using current result as the initial values of the next iteration
            if type in [1,2]:
                start_B = fitresult[0]
                start_tau = fitresult[1]
            print(fitresult)
            # Plot the fitting result
            plot_fit_and_data(t_fitting_data, y_fitting_data, fitresult, offset, type)
            # Define a0, a1, b0, b1
            a0 = 1
            a1, b0, b1 = calculate_coefficients(fitresult, fs, type)
            # Append the new coefficients to the existing IIR_coeffi list
            if i == num_iteration:
                print(type)
                print(fitresult)
                IIR_coeffi.append([a0, b0, a1, b1])

        # plot_fit_and_data(t_fitting_data, y_fitting_data, fitresult)
        
    write_in_csv(IIR_coeffi,IIR)

    # FIR part in under pending
    return Filter(fs, IIR, FIR)

def predistor(waveform, filter_sequence, fs):
    _filter = get_inverse_filter(waveform, filter_sequence, fs)
    return _filter.filering(waveform)
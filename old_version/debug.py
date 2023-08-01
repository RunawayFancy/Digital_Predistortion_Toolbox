import numpy as np
import predistortion_new as pd
import matplotlib.pyplot as plt
import csv
import pickle

def voltage_time_figure_plot(X1, Y1, color_code, name):
    # Create figure
    figure1 = plt.figure(name)
    
    # Create axes
    axes1 = figure1.add_subplot(1, 1, 1)
    
    # Create plot
    axes1.plot(X1, Y1, linewidth=2, color=color_code)
    
    # Set ylabel
    axes1.set_ylabel('Voltage (mV)', fontsize=16.5)
    
    # Set xlabel
    axes1.set_xlabel('Time (s)', fontsize=16.5)
    
    # Set title
    axes1.set_title(name, fontsize=16.5)
    
    # Set x-axis limit
    # axes1.set_xlim([0, 0.000001])
    
    # Uncomment the following line to set y-axis limit
    # axes1.set_ylim([-60, 60])
    
    # Uncomment the following line to set z-axis limit
    # axes1.set_zlim([-1, 1])
    
    axes1.grid(True)
    axes1.set_axisbelow(True)
    axes1.tick_params(labelsize=15)
    
    # Show the plot
    plt.show()

def write_list_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for item in data:
            writer.writerow([item])

def import_csv(filename):
    data_c1, data_c2 = [], []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the first line
        for row in csvreader:
            if len(row) >= 2:  # Ensure there are at least 2 columns in the row
                data_c1.append(float(row[0]))
                data_c2.append(float(row[1]))
    return [data_c1, data_c2]

def import_csv_1D(filename):
    data = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the first line
        for row in csvreader:
            data.append(float(row[0]))
    return data

def main():
    # ## Square pulse generating parameters
    # T = 50e-6  # pulse duration
    # T_enlong = 40e-6  # head and tail part of the square pulse
    # T_delay = 10e-6
    # N = int(500e3)  # number of sampled points - 1
    # Ts = (T + T_delay + T_enlong) / N  # sampling period
    # V_source = 1  # Amplitude of the step source, unit in V

    # t = np.linspace(0, T + T_enlong + T_delay, N + 1)  # unit in second
    # y = np.zeros(N + 1)
    # n_start = int(np.ceil(T_delay / Ts)) + 1
    # n_end = int(np.floor((T + T_delay) / Ts))
    # y[n_start-1:n_end] = V_source
    
    # # write_list_to_csv(y, 'Y.csv')

    # # voltage_time_figure_plot(t,y,'blue','Original square pulse')
    # ## Digital filtering parameters
    # type = 4  # 1 = RC high-pass; 2 = RC low-pass; 3 = RL high-pass; 4 = RL low-pass
    # L = 1e-6  # Inductance unit: H
    # C = 5e-7  # Capacitance unit: F
    # R = 10  # Resistance unit: Ohm

    # R_in = 0  # Internal resistance, unit: Ohm
    # R_total = R + R_in  # total resistance.

    # # Select desired filter type
    # if type == 1:
    #     # RC high-pass
    #     A = 0
    #     B = R * V_source / R_total
    #     tau = R_total * C
    # elif type == 2:
    #     # RC low-pass
    #     A = V_source
    #     B = -V_source
    #     tau = R_total * C
    # elif type == 3:
    #     # RL high-pass
    #     A = 0
    #     B = V_source
    #     tau = L / R_total
    # elif type == 4:
    #     # RL low-pass
    #     A = V_source * R / R_total
    #     B = -V_source * R / R_total
    #     tau = L / R_total

    # # Calculate the filter coefficient
    # zeta = Ts + 2 * tau
    # b0 = (2 * tau * A + 2 * tau * B + A * Ts) / zeta
    # b1 = (-2 * tau * A - 2 * tau * B + A * Ts) / zeta
    # a1 = (2 * tau - Ts) / zeta

    # y_new = []
    # # For 1st order filtering
    # for i in range(len(t)):
    #     if i == 0:
    #         y_new.append(b0 * y[i])
    #     else:
    #         y_new.append(b0 * y[i] + b1 * y[i-1] + a1 * y_new[i-1])

    # write_list_to_csv(y_new, 'Y_new.csv')
    # write_list_to_csv(t, 'X.csv')

    # voltage_time_figure_plot(t,y_new,'green', 'Distorted square pulse')

    # waveform = import_csv('temp.csv')

    # X = import_csv_1D('X.csv')
    # Y = import_csv_1D('Y.csv')
    X,Y =  pickle.load(open("E:/OP/Experiments/0609_predistortion/DATA/test/test/retest_500smooth_BareAWG.pkl","rb"))
    # X,Y =  pickle.load(open("0505_2023_144bits_T4/DATA/Predistor/test_500smooth_TLineA32_low_high_line_new.pkl","rb"))
    waveform = [X[0],np.mean(Y,axis=0)]
    waveform[0] = waveform[0]*10**(6)
    # waveform = [X,Y]

    amp_offset = waveform[1][0]*1
    waveform[1] = np.array(waveform[1]) - amp_offset
    gain = np.max(waveform[1])*0.99
    waveform[1] = waveform[1]/gain
    waveform_original = np.array(waveform)
    # type = 2
    filter_sequence = [2]   
    Ts = waveform[0][1]-waveform[0][0]
    fs = 1/Ts

    voltage_time_figure_plot(waveform_original[0],waveform_original[1],'green', 'Distorted square pulse')
    filted_waveform = np.array(pd.predistor(waveform, filter_sequence, fs))
    voltage_time_figure_plot(np.array(filted_waveform[0]),np.array(filted_waveform[1]), 'red', 'predistorted result')

    # filter_sequence_2nd = [2]
    # filted_waveform_2nd = np.array(pd.predistor(waveform, filter_sequence_2nd, fs))
    
    pickle.dump([filted_waveform[0],filted_waveform[1]] , open("filted.pkl" , "wb"))
    # Plotting
    

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(waveform[0], gain*np.array(filted_waveform[1]), color='blue', label='Filtered Waveform')

    ax.plot(waveform_original[0], gain*np.array(waveform_original[1]), color='red', label='Original Waveform')

    # ax.plot(filted_waveform_2nd[0], gain*np.array(filted_waveform_2nd[1]), color='orange', label='2nd Filtered Waveform')
    
    ax.grid()

    # Labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.set_title('Original vs. Filtered waveform')

    # Legend
    ax.legend()

    # Display the plot
    plt.show()

    # print(pd.calculate_coefficients([0,1, 1.43*10**(-7)], fs, 1))

# Check if the script is being run directly
if __name__ == "__main__":
    main()

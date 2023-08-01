import numpy as np
import pickle
import matplotlib.pyplot as plt

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

def set_mean(input_array):
    return np.mean(input_array, axis=0)
def set_offset(input_array, adjust):
    return np.array(input_array[1]) - input_array[1][0] * adjust
def set_gain(input_array, adjust):
    gain = np.abs(np.max(input_array[1]) * adjust)
    return np.array(input_array[1]) / gain

def moving_agerage(waveform,window_size ):
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []
    arr = np.array(waveform[1])
    # Loop through the array t o
    while i < len(arr) - window_size + 1:
    
        # Calculate the average of current window
        window_average = np.sum(arr[
        i:i+window_size]) / window_size
        
        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)
        
        # Shift window to right by one position
        i += 1
    waveform[0] = np.array(waveform[0])[0 : len(arr) - window_size+1]
    waveform[1] = moving_averages
    
    return waveform

waveform = []
waveform_atten  = []

index = []
index_atten = []

window_size = 500


character_set = ["A", "B", "C", "D", "E", "F"]
length = len(character_set)
for index_chara1 in range(length):
    chara1= character_set[index_chara1]
    file_name = "E:/OP/Experiments/0616_2023_pred/DATA/test/copper_line/Line"+ chara1 + chara1 +"_10rep_80G_80kpts_70us_TLineCase2.pkl"
    waveform_dummy = moving_agerage(pickle.load(open(file_name, "rb")), window_size)
    waveform_dummy[1] = set_offset(waveform_dummy,1)
    waveform_dummy[1] = set_gain(waveform_dummy, 1)
    index.append(chara1 + chara1)
    waveform.append(waveform_dummy)
    if chara1 not in ["A", "F"]:
        file_name = "E:/OP/Experiments/0616_2023_pred/DATA/test/copper_line/Line"+ chara1 + chara1 +"_10dbAtten_10rep_80G_80kpts_70us_TLineCase2.pkl"
        waveform_dummy = moving_agerage(pickle.load(open(file_name, "rb")), window_size)
        waveform_dummy[1] = set_offset(waveform_dummy,1)
        waveform_dummy[1] = set_gain(waveform_dummy, 1)
        index_atten.append(chara1 + chara1)
        waveform_atten.append(waveform_dummy)



waveform_AWG =  pickle.load(open("E:/OP/Experiments/0616_2023_pred/DATA/test/copper_line/BareAWGT_10rep_80G_80kpts_70us_LineCase2.pkl", "rb"))# moving_agerage(pickle.load(open("E:/OP/Experiments/0616_2023_pred/DATA/retest_500smooth_BareAWG_new_averaged.pkl", "rb")), window_size)
waveform_AWG[1] = set_offset(np.copy(waveform_AWG),1)
waveform_AWG[1] = set_gain(np.copy(waveform_AWG),1)
waveform_AWG = waveform_AWG

print(np.shape(waveform_AWG[0]))

index_L0 = ['AA', 'BB', 'CC', 'DD', 'EE', 'FF']
index_L1 = ['AB', 'BC', 'CD', 'DE', 'EF']
index_L2 = ['AC', 'BD', 'CE', 'DF']
index_L3 = ['AD', 'BE', 'CF']
index_L4 = ['AE', 'BF']
index_L5 = ['AF']

index_want_plot =  ['BF']# ['AE', 'BE', 'CE', 'DE', 'EF'] # ['AD', 'BD', 'CD', 'DE', 'DF'] # [ 'AB', 'AC', 'AD', 'AE', 'AF'] # ['AB', 'BC', 'BD', 'BE', 'BF'] #['AC','BC', 'CD', 'CE', 'CF'] # ['AF', 'BF', 'CF', 'DF', 'EF']

X = np.array(waveform[0][0]) * 10**6

fig = plt.figure()
ax = fig.add_subplot(111)

# for i in range(len(index)):
#     if index[i] not in index_L0:
#         ax.plot(X, waveform[i][1], label='Line '+ index[i])
# ax.grid()

for line_code in index_want_plot:
    i = np.where( np.array(index) == line_code)[0][0]
    ax.plot(X, waveform[i][1], label='Line '+ index[i])
ax.grid()

ax.plot(waveform_AWG[0], waveform_AWG[1], label='AWG_Bare')

# Labels and title
ax.set_xlabel('Time (us)')
ax.set_ylabel('Amplitude')
ax.set_title('Distorted waveform on copper lines, cable = 2.5m')

# Legend
ax.legend()

# Display the plot
plt.show()

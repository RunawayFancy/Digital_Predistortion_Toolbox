import numpy as np
import matplotlib.pyplot as plt
import pickle

def get_mean(input_array):
    return np.mean(input_array, axis=0)
def set_offset(input_array):
    return np.array(input_array) - input_array[0]

# X,Y_bare =  pickle.load(open("0505_2023_144bits_T4/DATA/Predistor/test_500smooth_TLineA32.pkl","rb"))
# X_high2_line,Y_high2_line =  pickle.load(open("0505_2023_144bits_T4/DATA/Predistor/test_500smooth_TLineA32_high2_line_new.pkl","rb"))
# X_high2,Y_high2 =  pickle.load(open("0505_2023_144bits_T4/DATA/Predistor/test_500smooth_TLineA32_high2_new.pkl","rb"))
# X_high_line,Y_high_line =  pickle.load(open("0505_2023_144bits_T4/DATA/Predistor/test_500smooth_TLineA32_high_line_new.pkl","rb"))
# X_low_high_line,Y_low_high_line =  pickle.load(open("0505_2023_144bits_T4/DATA/Predistor/test_500smooth_TLineA32_low_high_line_new.pkl","rb"))
# X_low,Y_low =  pickle.load(open("0505_2023_144bits_T4/DATA/Predistor/test_500smooth_TLineA32_low_new.pkl","rb"))
# X_low_high,Y_low_high =  pickle.load(open("0505_2023_144bits_T4/DATA/Predistor/test_500smooth_TLineA32_low_high.pkl","rb"))

X, Y_AC = pickle.load(open("E:/OP/Experiments/0609_predistortion/DATA/test/test/retest_500smooth_CopperLineAC_TLineCase2.pkl","rb"))
X, Y_AF = pickle.load(open("E:/OP/Experiments/0609_predistortion/DATA/test/test/retest_500smooth_CopperLineAF_TLineCase2.pkl","rb"))
X, Y_BC = pickle.load(open("E:/OP/Experiments/0609_predistortion/DATA/test/test/retest_500smooth_CopperLineBC_TLineCase2.pkl","rb"))
X, Y_BF = pickle.load(open("E:/OP/Experiments/0609_predistortion/DATA/test/test/retest_500smooth_CopperLineBF_TLineCase2.pkl","rb"))
X, Y_CF = pickle.load(open("E:/OP/Experiments/0609_predistortion/DATA/test/test/retest_500smooth_CopperLineCF_TLineCase2.pkl","rb"))
X, Y_DF = pickle.load(open("E:/OP/Experiments/0609_predistortion/DATA/test/test/retest_500smooth_CopperLineDF_TLineCase2.pkl","rb"))
X, Y_EF = pickle.load(open("E:/OP/Experiments/0609_predistortion/DATA/test/test/retest_500smooth_CopperLineEF_TLineCase2.pkl","rb"))
X, Y_AWG = pickle.load(open("E:/OP/Experiments/0609_predistortion/DATA/test/test/retest_500smooth_BareAWG.pkl","rb"))

Y_AC = set_offset( get_mean(Y_AC) )
Y_AF = set_offset( get_mean(Y_AF) )
Y_BC = set_offset( get_mean(Y_BC) )
Y_BF = set_offset( get_mean(Y_BF) )
Y_CF = set_offset( get_mean(Y_CF) )
Y_DF = set_offset( get_mean(Y_DF) )
Y_EF = set_offset( get_mean(Y_EF) )
Y_AWG = set_offset( get_mean(Y_AWG) )


fig = plt.figure()
ax = fig.add_subplot(111)
# # ax.plot(X,Y_combined, color='blue', label='AWG + Line')
# ax.plot(X[0], np.mean(Y_low_high_line,axis=0), label='Lowpass+Highpass+Linear')
# ax.plot(X[0], np.mean(Y_high2_line, axis=0), label='Highpass+Linear')
# ax.plot(X[0], np.mean(Y_bare, axis=0), label='T.L. Distorted')
# # ax.plot(X[0], np.mean(Y_low_high, axis=0), label='Lowpass + Highpass')
# ax.plot(X[0], np.mean(Y_low, axis=0), label='Lowpass')
# ax.plot(X[0], np.mean(Y_high2, axis=0), label='Highpass')
# #ax.plot(X[0], Y_combined, color='blue', label='Lowpass+Highpass')

# ax.plot(X_test1[0], Y1, label='Test1 T.Line = 3 m')
# ax.plot(X_test1[0], Y2, label='Test2 T.Line = 2 m')
# ax.plot(X_test1[0], Y3, label='Test3 T.Line = 2.5 m')

ax.plot(X[0], Y_AF, label='Line AF Cable = 2.5 m')
ax.plot(X[0], Y_AC, label='Line AC Cable = 2.5 m')
# ax.plot(X[0], Y_BC, label='Line BC Cable = 2.5 m')
# ax.plot(X[0], Y_BF, label='Line BF Cable = 2.5 m')
# ax.plot(X[0], Y_CF, label='Line CF Cable = 2.5 m')
# ax.plot(X[0], Y_DF, label='Line DF Cable = 2.5 m')
# ax.plot(X[0], Y_EF, label='Line EF Cable = 2.5 m')

ax.plot(X[0], Y_AWG, label='Bare AWG')

ax.grid()

    # Labels and title
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')
ax.set_title('Distorted vs. Filtered Waveform')

    # Legend
ax.legend()

    # Display the plot
plt.show()
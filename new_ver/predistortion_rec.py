import numpy as np
import csv
from math import factorial


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

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')
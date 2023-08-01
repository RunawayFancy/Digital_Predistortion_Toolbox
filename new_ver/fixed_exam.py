import numpy as np

class smooth_parameter:
    """
    The waveform should be normalized to one
    """

    def __init__(self,waveform,param):
        """
        waveform: 2D arrray/list
        param = [start_time, end_time, windows_size, ref_line_pos]
        """
        self.waveform = waveform
        self.start_time = param[0]
        self.ref_line_position = param[3]
        self.start_point = np.where(np.array(self.waveform[0]) > self.start_time)[0][0]
        self.end_time = param[1]
        self.end_point = np.where(np.array(self.waveform[0]) > self.end_time)[0][0] 
        self.windows_size = param[2]
        self.ref_line = self.get_ref_line()

    def get_ref_line(self):
        if self.ref_line_position == "start":
            amp_segment = self.waveform[1][self.start_point:(self.start_point+self.windows_size)]
        else:
            amp_segment = self.waveform[1][(self.end_point-self.windows_size):self.end_point]
        return np.mean(amp_segment)
    
    def calculate_smooth_param(self):
        amp_segment = self.waveform[1][self.start_point:self.end_point]
        smooth_parameter = 0
        for amp in amp_segment:
            smooth_parameter += (amp - self.ref_line)**2
        return np.sqrt(smooth_parameter)

def smooth_param(waveform,param):
    _sp = smooth_parameter(waveform,param)
    return _sp.calculate_smooth_param()
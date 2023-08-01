from scipy.special import erfc
import numpy as np

def Line_response(time, fitresult, param):
    _, _, _, _, time_offset, amp_offset = param
    coef_B = fitresult[0]
    return amp_offset + coef_B * (np.array(time) - time_offset)

def lowhigh_pass_reponse(time, fitresult, param):
    _, _, _, _, time_offset, amp_offset = param
    coef_B, coef_tau = fitresult
    return amp_offset + coef_B * np.exp(- (np.array(time) - time_offset ) / coef_tau)

def skin_response(time, fitresult, param):
    _, _, _, _, time_offset, amp_offset = param
    coef_B = fitresult[0]
    return amp_offset + erfc(coef_B/(np.sqrt(np.array(time) - time_offset)))
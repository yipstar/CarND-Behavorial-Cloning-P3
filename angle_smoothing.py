import numpy as np
from pandas import ewma

def perform_angle_smoothing(y_data):
    angles_array = np.asarray(y_data)
    fwd = ewma(angles_array, span=20)
    bwd = ewma(angles_array[::-1], span=20)
    smooth = np.vstack((fwd, bwd[::-1]))
    smooth = np.mean(smooth, axis=0)
    return smooth
    # angles = np.ndarray.tolist(smooth)
    # return angles

import numpy as np
from .objective import sum_data_rate

class Antenna:
    def __init__(self, pos):
        self.pos = np.array(pos)

class UAV:
    def __init__(self, uid, pos):
        self.uid = uid
        self.pos = np.array(pos)

def evaluate(uavs, antennas, theta=None, K=1.0):
    uav_positions = {u.uid: tuple(u.pos) for u in uavs}
    antenna_list = [{'pos': tuple(a.pos)} for a in antennas]
    return sum_data_rate(uav_positions, antenna_list, theta=theta, K=K)

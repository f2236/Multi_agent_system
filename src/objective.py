import math
import numpy as np

def free_space_path_loss(dist, freq_hz=2.4e9):
    if dist <= 0:
        return 1e-12
    c = 3e8
    lam = c / freq_hz
    return (4 * math.pi * dist / lam) ** 2

def rate_between(tx_pos, rx_pos, tx_power_dbm=20.0, bandwidth_hz=1e6, noise_figure_db=7.0, freq_hz=2.4e9):
    # Simple rate model: r = B * log2(1 + SNR)
    # Convert powers
    tx_power_w = 10 ** ((tx_power_dbm - 30) / 10)
    dist = np.linalg.norm(np.array(tx_pos) - np.array(rx_pos)) + 1e-6
    path_loss = free_space_path_loss(dist, freq_hz)
    # assume unit gains
    rx_noise_w = 1e-9 * 10 ** (noise_figure_db / 10)
    snr = tx_power_w / (path_loss * rx_noise_w)
    rate = bandwidth_hz * math.log2(1 + snr)
    return rate

def sum_data_rate(uav_positions, antenna_list, theta=None, K=1.0):
    """
    uav_positions: dict user_id -> 3D position (x,y,z)
    antenna_list: one communication/FAS port dict with 'pos' key
    theta: dict (antenna_index, user_id) -> weight (defaults to 1)
    K: antenna-port count fixed to 1 for the client/paper setup
    """
    if float(K) != 1.0:
        raise ValueError("Only K=1 is supported; antenna-port selection is fixed by Equation 23.")
    if len(antenna_list) != 1:
        raise ValueError("Expected exactly one communication/FAS port when K=1.")

    total = 0.0
    for u_id, u_pos in uav_positions.items():
        for m_idx, ant in enumerate(antenna_list):
            w = 1.0
            if theta is not None:
                w = theta.get((m_idx, u_id), 1.0)
            r = rate_between(u_pos, ant['pos'])
            total += w * r
    return total

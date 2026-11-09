"""Reusable filters for magnetic-field time series."""

import numpy as np
from scipy.signal import butter, filtfilt


def butterworth_filter(data, cutoff_freq, fs, order=4):
    """Apply a low-pass Butterworth filter to estimate background field."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, np.asarray(data))


def compute_delta_bt_from_total(bt, cutoff_freq, fs, order=4):
    """Compute background total field and fluctuation from total field."""
    bt_arr = np.asarray(bt)
    bt_bg = butterworth_filter(bt_arr, cutoff_freq=cutoff_freq, fs=fs, order=order)
    delta_bt = bt_arr - bt_bg
    return bt_bg, delta_bt

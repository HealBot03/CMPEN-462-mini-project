# run from project root: python test_real_audio.py
import numpy as np
from scipy.signal import find_peaks

def compute_distance_textbook(rx_block, preamble_block, fs=48000):
    """
    Strict implementation of the FFT -> Divide -> iFFT pipeline.
    """
    N = len(preamble_block)
    if len(rx_block) < N:
        rx_block = np.pad(rx_block, (0, N - len(rx_block)))
    else:
        rx_block = rx_block[:N]

    # 1 fft
    X = np.fft.fft(preamble_block)
    Y = np.fft.fft(rx_block)
    
    # 2 division deconvolution
    # divide step in frequency domain
    H = Y / (X + 0.05 * np.max(np.abs(X)))
    
    # 3 ifft for channel impulse response
    h = np.fft.ifft(H)
    mag = np.abs(h)
    
    # 4 peak detection
    # minimum height threshold to reduce noise ripples
    peaks, _ = find_peaks(mag, height=np.max(mag)*0.2)
    
    if len(peaks) < 2:
        return None, mag
        
    # pick indices of the two strongest peaks
    idx = peaks[np.argsort(mag[peaks])[-2:]]
    n1, n2 = idx[0], idx[1]
    
    # 5 circular gap
    # handle wrap around past index 0
    diff = abs(n2 - n1)
    gap = min(diff, len(mag) - diff)
    
    # 6 physical conversion
    # c = 343.0 m/s speed of sound
    c = 343.0
    extra_path = c * gap / fs
    wall_distance = extra_path / 2.0
    
    return wall_distance, mag
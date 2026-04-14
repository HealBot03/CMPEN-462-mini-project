import numpy as np
from scipy.io import wavfile

def create_preamble():
    # parameters
    fs = 48000  # sample rate in Hz 
    
    # 4096 samples of Gaussian random noise
    # roughly 85ms burst
    p = np.random.randn(4096) 
    
    # normalize to prevent audio clipping (-1.0 to 1.0)
    p_normalized = p / np.max(np.abs(p))
    
    # convert to 32 bit .wav
    p_wav = np.float32(p_normalized)
    
    wavfile.write("preamble.wav", fs, p_wav)
    print("preamble.wav generated successfully!")

if __name__ == "__main__":
    create_preamble()
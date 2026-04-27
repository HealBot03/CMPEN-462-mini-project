# import standard math libraries
import numpy as np
from scipy.signal import find_peaks

# physical constant
SPEED_OF_SOUND_M_S = 343.0

# calculate wall distance using fft deconvolution
def compute_distance(
    rx_block: np.ndarray,
    preamble_block: np.ndarray,
    fs: int = 48_000,
    epsilon_scale: float = 0.05,
) -> tuple[float | None, np.ndarray, tuple[int, int] | None, tuple[float, float] | None, int | None]:
    """
    FFT -> Divide -> iFFT

    Returns:
    wall_distance_m: estimated one way distance in meters, or None when two peaks are unavailable
    cir_magnitude: magnitude of the circular channel impulse response
    peak_indices: indices of the two strongest magnitude samples
    peak_magnitudes: magnitudes at strongest
    circular_gap_samples: sample gap using circular distance on block length N
    """
    # math: cast incoming arrays to standard float format
    preamble = np.asarray(preamble_block, dtype=np.float32)
    rx = np.asarray(rx_block, dtype=np.float32)

    # matches the preamble block size
    # get fundamental block length
    N = int(len(preamble))
    if N <= 1:
        return None, np.zeros(0, dtype=np.float32), None, None, None

    # truncate receiver block to match preamble length N
    # math: enforce equal array lengths for fft
    if len(rx) < N:
        rx = np.pad(rx, (0, N - len(rx)))
    else:
        rx = rx[:N]

    # 1) FFT of known preamble
    # math: fast fourier transform of reference
    X = np.fft.fft(preamble, n=N)

    # 2) FFT of recorded audio block
    # math: fast fourier transform of recorded signal
    Y = np.fft.fft(rx, n=N)

    # 3) division with scaled epsilon floor to prevent noise explosion
    # add noise floor proportional to max peak
    epsilon = float(epsilon_scale) * float(np.max(np.abs(X))) + 1e-12
    # math: calculate transfer function h = y / (x + epsilon)
    H = Y / (X + epsilon)

    # 4) Inverse FFT to obtain circular channel impulse response (CIR)
    # invert to time domain
    h = np.fft.ifft(H, n=N)
    # strip phase and keep absolute magnitude
    magnitude = np.abs(h).astype(np.float32, copy=False)

    if N < 2:
        return None, magnitude, None, None, None

    # 5) Peak detection
    # 'distance=30' ensures peaks must be at least ~10cm apart at 48kHz
    # preventing the selection of adjacent samples on a single wide peak
    # find local maxima
    peaks, _ = find_peaks(magnitude, distance=30)

    if len(peaks) < 2:
        # not enough distinct paths detected in this block
        return None, magnitude, None, None, None

    # identify the two strongest distinct peaks
    # sort by height and keep indices of top two
    peak_heights = magnitude[peaks]
    top_two_order = np.argsort(peak_heights)[-2:]
    peak_indices = peaks[top_two_order]

    n1 = int(peak_indices[0])
    n2 = int(peak_indices[1])
    m1 = float(magnitude[n1])
    m2 = float(magnitude[n2])

    # calculate circular sample gap to handle CIR rotation
    # math: compute shortest index distance in circular array
    diff = abs(n2 - n1)
    circular_gap = min(diff, N - diff)

    # convert sample delay to physical one-way wall distance
    # total path difference Delta L = c * (Delta n / fs)
    # distance = speed * (gap / fs) / 2
    wall_distance = SPEED_OF_SOUND_M_S * circular_gap / float(fs) / 2.0

    return wall_distance, magnitude, (n1, n2), (m1, m2), int(circular_gap)
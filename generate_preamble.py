import argparse
import numpy as np
from scipy.io import wavfile

# constants
DEFAULT_SAMPLE_RATE = 48000
BASE_PREAMBLE_SAMPLES = 4096
PREAMBLE_REPEATS = 4

def float_to_int16(audio: np.ndarray) -> np.ndarray:
    """converts floating point audio (-1.0 to 1.0) to 16-bit PCM."""
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)

def generate_preamble(
    base_samples: int = BASE_PREAMBLE_SAMPLES,
    repeats: int = PREAMBLE_REPEATS,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """gnerates a Gaussian base block and a tiled version for playback"""
    
    # build the base 4096-sample random block
    np.random.seed(seed)
    base = np.random.randn(base_samples).astype(np.float32)
    
    # normalize peak to 1.0 to prevent clipping
    peak = float(np.max(np.abs(base)))
    if peak >= 1e-12:
        base = (base / peak).astype(np.float32)

    # tile the block for continuous phone playback
    phone_signal = np.tile(base, repeats)
    return base, phone_signal

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="generate 4x tiled Gaussian preamble WAV.")
    parser.add_argument("--samplerate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--output", default="preamble.wav")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    # generate the signal
    base_block, phone_signal = generate_preamble()
    
    # convert to 16-bit PCM and save
    pcm_signal = (np.clip(phone_signal, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(args.output, args.samplerate, pcm_signal)
    
    print(f"Saved {args.output} | fs={args.samplerate}Hz | Base: {len(base_block)} | Tiled: {len(phone_signal)}")

if __name__ == "__main__":
    main()
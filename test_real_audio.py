# import required libraries for type hinting and timing
from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass

# import external math and plotting libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

# import local signal processing modules
from generate_preamble import (
    BASE_PREAMBLE_SAMPLES,
    DEFAULT_SAMPLE_RATE,
    PREAMBLE_REPEATS,
    generate_preamble,
    float_to_int16,
)
from signal_math import compute_distance

# try to load audio device library
try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - handled at runtime
    sd = None


# hold echo estimation results
@dataclass
class EchoEstimate:
    distance_m: float
    circular_gap_samples: int
    peak_indices: tuple[int, int]
    peak_magnitudes: tuple[float, float]
    cir: np.ndarray


# setup and return command line arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate reflector distance from live microphone audio or a WAV file "
            "using block-based FFT deconvolution."
        )
    )
    parser.add_argument(
        "--input-file",
        help="Optional WAV file to analyze instead of the live microphone.",
    )
    parser.add_argument(
        "--device",
        help="Input device index or device name for sounddevice.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available audio devices and exit.",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Input sample rate in Hz.",
    )
    parser.add_argument(
        "--block-samples",
        type=int,
        default=BASE_PREAMBLE_SAMPLES,
        help="Processing block size in samples. Use 4096 to match the CMPEN pipeline.",
    )
    parser.add_argument(
        "--preamble-repeats",
        type=int,
        default=PREAMBLE_REPEATS,
        help="How many times to tile the 4096-sample base block for phone playback.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Optional live capture duration in seconds. Ctrl+C also stops the script.",
    )
    parser.add_argument(
        "--probe-seed",
        type=int,
        default=7,
        help="Random seed used for Gaussian white-noise base block generation.",
    )
    parser.add_argument(
        "--epsilon-scale",
        type=float,
        default=0.05,
        help="Scaled epsilon in H=Y/(X+epsilon), where epsilon=epsilon_scale*max(|X|).",
    )
    parser.add_argument(
        "--trigger-peak-ratio",
        type=float,
        default=10.0,
        help="Live mode trigger threshold: strongest CIR peak must be this multiple of median CIR noise floor.",
    )
    parser.add_argument(
        "--trigger-trials",
        type=int,
        default=0,
        help="Number of valid peak-triggered trials before freeze; use 0 for infinite trials.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Median filter length applied to the printed distance.",
    )
    parser.add_argument(
        "--phone-preamble-out",
        default="preamble.wav",
        help="Mono WAV output path for the 4x tiled phone playback preamble.",
    )
    parser.add_argument(
        "--save-plot",
        default="cir_live_snapshot.png",
        help="Path used to save the last valid CIR snapshot.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable the live/debug plot.",
    )
    return parser.parse_args()


# convert audio to mono floating point array and remove dc offset
def to_mono_float(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio[:, 0]

    if np.issubdtype(audio.dtype, np.integer):
        scale = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / float(scale)
    else:
        audio = audio.astype(np.float32, copy=False)

    # math: normalize by subtracting the mean
    return (audio - np.mean(audio)).astype(np.float32, copy=False)


# parse device id or name for audio stream
def resolve_device(device: str | None) -> int | str | None:
    if device is None:
        return None
    try:
        return int(device)
    except ValueError:
        return device


# generate reference audio and save to disk
"""
def load_reference(args: argparse.Namespace) -> tuple[int, np.ndarray, np.ndarray]:
    fs = int(args.samplerate)
    block_samples = max(1, int(args.block_samples))
    preamble_repeats = max(1, int(args.preamble_repeats))

    base_block, phone_signal = generate_preamble(
        base_samples=block_samples,
        repeats=preamble_repeats,
        seed=args.probe_seed,
    )

    wavfile.write(args.phone_preamble_out, fs, float_to_int16(phone_signal))
    print(f"Exported phone playback preamble to {args.phone_preamble_out}")

    return fs, base_block, phone_signal
    """

def load_reference(args: argparse.Namespace) -> tuple[int, np.ndarray, np.ndarray]:
    fs = int(args.samplerate)
    block_samples = max(1, int(args.block_samples))
    # ensure 'preamble_external.wav' is in your project folder
    try:
        file_fs, external_audio = wavfile.read("preamble_external.wav")
    except FileNotFoundError:
        raise FileNotFoundError("could not find preamble_external.wav")

    # convert to mono float and remove dc offset
    full_preamble = to_mono_float(external_audio)

    # equires the base_block to match the fft block size N
    base_block = full_preamble[:block_samples] 
    phone_signal = full_preamble 

    print(f"loaded external preamble | fs={file_fs}hz | total samples={len(phone_signal)}")
    
    return fs, base_block, phone_signal


# process block to find distance using signal math
def estimate_distance(
    audio_block: np.ndarray,
    preamble_block: np.ndarray,
    fs: int,
    epsilon_scale: float,
) -> EchoEstimate | None:
    block = to_mono_float(audio_block)
    if np.max(np.abs(block)) < 1e-5:
        return None

    distance_m, cir, peak_indices, peak_magnitudes, circular_gap = compute_distance(
        rx_block=block,
        preamble_block=preamble_block,
        fs=fs,
        epsilon_scale=epsilon_scale,
    )
    if (
        distance_m is None
        or peak_indices is None
        or peak_magnitudes is None
        or circular_gap is None
    ):
        return None

    return EchoEstimate(
        distance_m=distance_m,
        circular_gap_samples=circular_gap,
        peak_indices=peak_indices,
        peak_magnitudes=peak_magnitudes,
        cir=cir,
    )


# math: calculate signal to noise ratio of highest peak
def compute_peak_to_noise_ratio(estimate: EchoEstimate) -> float:
    noise_floor = float(np.median(estimate.cir)) + 1e-12
    peak = float(max(estimate.peak_magnitudes))
    return peak / noise_floor


# initialize interactive matplotlib figure
def create_plot() -> tuple[plt.Figure, plt.Axes, object, tuple[object, object]]:
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    line, = ax.plot([], [], linewidth=2)
    peak1_marker = ax.axvline(0.0, color="tab:green", linestyle="--", linewidth=1.5, label="Peak 1")
    peak2_marker = ax.axvline(0.0, color="tab:red", linestyle="--", linewidth=1.5, label="Peak 2")
    ax.set_title("Block Deconvolution CIR Magnitude")
    ax.set_xlabel("CIR sample index (circular)")
    ax.set_ylabel("Magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, ax, line, (peak1_marker, peak2_marker)


# update existing plot with new data
def update_plot(
    ax: plt.Axes,
    line,
    markers,
    estimate: EchoEstimate,
    smoothed_distance_m: float,
) -> None:
    axis_samples = np.arange(len(estimate.cir), dtype=np.int32)
    line.set_data(axis_samples, estimate.cir)
    markers[0].set_xdata([estimate.peak_indices[0], estimate.peak_indices[0]])
    markers[1].set_xdata([estimate.peak_indices[1], estimate.peak_indices[1]])

    ax.set_xlim(0, max(1, len(estimate.cir) - 1))
    ax.set_ylim(0.0, float(np.max(estimate.cir)) * 1.1)
    ax.set_title(
        "Block Deconvolution CIR Magnitude "
        f"(instant={estimate.distance_m:.3f} m, smooth={smoothed_distance_m:.3f} m, "
        f"gap={estimate.circular_gap_samples} samples)"
    )
    ax.figure.canvas.draw_idle()
    plt.pause(0.001)


# export single frame of plot to file
def save_snapshot(output_path: str, estimate: EchoEstimate | None) -> None:
    if estimate is None:
        return

    fig, ax = plt.subplots(figsize=(10, 4.5))
    axis_samples = np.arange(len(estimate.cir), dtype=np.int32)

    ax.plot(axis_samples, estimate.cir, linewidth=2)
    ax.axvline(
        estimate.peak_indices[0],
        color="tab:green",
        linestyle="--",
        linewidth=1.5,
        label=f"Peak 1 ({estimate.peak_indices[0]})",
    )
    ax.axvline(
        estimate.peak_indices[1],
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label=f"Peak 2 ({estimate.peak_indices[1]})",
    )
    ax.set_title(
        f"Block Deconvolution CIR Snapshot (distance = {estimate.distance_m:.3f} m, "
        f"gap = {estimate.circular_gap_samples} samples)"
    )
    ax.set_xlabel("CIR sample index (circular)")
    ax.set_ylabel("Magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# read from wav file and compute distances block by block
def run_file_mode(
    audio_path: str,
    preamble_block: np.ndarray,
    fs: int,
    args: argparse.Namespace,
) -> EchoEstimate | None:
    file_fs, audio = wavfile.read(audio_path)
    if file_fs != fs:
        raise ValueError(
            f"Input file sample rate is {file_fs} Hz, but requested sample rate is {fs} Hz."
        )

    audio = to_mono_float(audio)
    block_samples = len(preamble_block)

    smoothing = deque(maxlen=max(1, args.smooth))
    accepted_estimates: list[tuple[float, EchoEstimate]] = []

    for start in range(0, len(audio), block_samples):
        block = audio[start : start + block_samples]
        if len(block) < block_samples:
            block = np.pad(block, (0, block_samples - len(block)))

        estimate = estimate_distance(
            block,
            preamble_block,
            fs,
            args.epsilon_scale,
        )
        if estimate is None:
            continue

        # math: apply median filter to distance values
        smoothing.append(estimate.distance_m)
        smooth_distance = float(np.median(np.asarray(smoothing)))
        print(
            f"block={start // block_samples:5d}  "
            f"t={start / fs:7.3f}s  "
            f"instant={estimate.distance_m:0.3f} m  "
            f"smooth={smooth_distance:0.3f} m  "
            f"gap={estimate.circular_gap_samples:5d} samples  "
            f"peaks={estimate.peak_indices}"
        )
        accepted_estimates.append((start / fs, estimate))

    if not accepted_estimates:
        print("No valid two-peak deconvolution response was detected in the WAV file.")
        return None

    distances = np.asarray([estimate.distance_m for _, estimate in accepted_estimates], dtype=np.float32)
    median_distance = float(np.median(distances))
    representative_time, representative_estimate = min(
        accepted_estimates,
        key=lambda item: (
            abs(item[1].distance_m - median_distance),
            -(item[1].peak_magnitudes[0] + item[1].peak_magnitudes[1]),
        ),
    )
    print(
        f"Median distance from file: {median_distance:.3f} m "
        f"(representative block at t={representative_time:.3f}s, "
        f"gap={representative_estimate.circular_gap_samples} samples, "
        f"peaks={representative_estimate.peak_indices})"
    )
    return representative_estimate


# read from live mic and compute distances
def run_live_mode(
    preamble_block: np.ndarray,
    fs: int,
    args: argparse.Namespace,
) -> EchoEstimate | None:
    if sd is None:
        raise RuntimeError(
            "Live mode requires sounddevice. Install it with: pip install sounddevice"
        )

    device = resolve_device(args.device)
    block_samples = len(preamble_block)
    trigger_trials = int(args.trigger_trials)
    unlimited_trials = trigger_trials <= 0

    last_estimate = None
    smoothing = deque(maxlen=max(1, args.smooth))
    triggers_captured = 0

    plot_state = None
    if not args.no_plot:
        plot_state = create_plot()

    print(
        f"Listening at {fs} Hz with block size {block_samples}. "
        f"Collecting {'infinite' if unlimited_trials else trigger_trials} triggered trial(s). "
        "Press Ctrl+C to stop."
    )
    if device is not None:
        print(f"Using input device: {device}")

    start_time = time.time()
    with sd.InputStream(
        samplerate=fs,
        blocksize=block_samples,
        channels=1,
        dtype="float32",
        device=device,
        latency="low",
    ) as stream:
        while True:
            if args.duration is not None and (time.time() - start_time) >= args.duration:
                break

            chunk, overflowed = stream.read(block_samples)
            block = np.asarray(chunk[:, 0], dtype=np.float32)

            if overflowed:
                print("\nWarning: input overflow detected. Try a larger audio buffer.")

            estimate = estimate_distance(
                block,
                preamble_block,
                fs,
                args.epsilon_scale,
            )
            if estimate is None:
                print("\rListening... awaiting valid two-peak block response.    ", end="", flush=True)
                continue

            peak_to_noise = compute_peak_to_noise_ratio(estimate)
            if peak_to_noise < float(args.trigger_peak_ratio):
                print(
                    "\rListening... peak below trigger "
                    f"({peak_to_noise:0.2f}x < {args.trigger_peak_ratio:0.2f}x).    ",
                    end="",
                    flush=True,
                )
                continue

            last_estimate = estimate
            triggers_captured += 1
            smoothing.append(estimate.distance_m)
            smooth_distance = float(np.median(np.asarray(smoothing)))
            print()
            print(
                f"Trial {triggers_captured}"
                f"{'/' + str(trigger_trials) if not unlimited_trials else ''} detected. "
                f"Distance={smooth_distance:0.3f} m  "
                f"instant={estimate.distance_m:0.3f} m  "
                f"gap={estimate.circular_gap_samples:5d} samples  "
                f"peaks={estimate.peak_indices}  "
                f"peak/noise={peak_to_noise:0.2f}x"
            )

            if plot_state is not None:
                _, ax, line, markers = plot_state
                update_plot(ax, line, markers, estimate, smooth_distance)

            if (not unlimited_trials) and triggers_captured >= trigger_trials:
                print("Reached requested number of triggered trials. Freezing on final result.")
                break

    if plot_state is not None and last_estimate is not None:
        _, ax, line, markers = plot_state
        final_smooth_distance = float(np.median(np.asarray(smoothing)))
        update_plot(ax, line, markers, last_estimate, final_smooth_distance)
        plt.ioff()
        plt.show(block=True)

    print()
    return last_estimate


# list all system audio devices
def print_device_list() -> None:
    if sd is None:
        raise RuntimeError("sounddevice is not installed, so devices cannot be listed.")
    print(sd.query_devices())


# handle main script execution
def main() -> None:
    args = parse_args()

    if args.list_devices:
        print_device_list()
        return

    fs, preamble_block, phone_signal = load_reference(args)
    print(
        "Generated Gaussian preamble base block and phone playback signal "
        f"(base={len(preamble_block)} samples, repeats={args.preamble_repeats}, "
        f"phone_total={len(phone_signal)} samples, seed={args.probe_seed}, fs={fs} Hz)"
    )

    best_estimate = None
    try:
        if args.input_file:
            best_estimate = run_file_mode(args.input_file, preamble_block, fs, args)
        else:
            best_estimate = run_live_mode(preamble_block, fs, args)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        save_snapshot(args.save_plot, best_estimate)
        if best_estimate is not None:
            print(f"Saved CIR snapshot to {args.save_plot}")


if __name__ == "__main__":
    main()
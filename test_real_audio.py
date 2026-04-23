from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - handled at runtime
    sd = None

# speed of sound used for sample to distance conversion

SPEED_OF_SOUND_M_S = 343.0


# output container for one accepted echo estimate
@dataclass
class EchoEstimate:
    distance_m: float
    direct_idx: int
    echo_idx: int
    direct_peak: float
    echo_peak: float
    echo_ratio: float
    echo_snr: float
    burst_snr: float
    cir: np.ndarray


def parse_args() -> argparse.Namespace:
    # parse cli options for live and file analysis
    parser = argparse.ArgumentParser(
        description=(
            "Estimate reflector distance from live microphone audio or a WAV file "
            "using the training sequence stored in preamble.wav."
        )
    )
    parser.add_argument(
        "--preamble",
        default="preamble.wav",
        help="Reference training sequence used for matched filtering.",
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
        help="Override the input sample rate. Defaults to the preamble sample rate.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=0.8,
        help="Rolling analysis window length in seconds.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=0.1,
        help="Microphone chunk length in seconds for live mode.",
    )
    parser.add_argument(
        "--step-seconds",
        type=float,
        default=0.2,
        help="Sliding step used when analyzing a WAV file.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Optional live capture duration in seconds. Ctrl+C also stops the script.",
    )
    parser.add_argument(
        "--lowcut",
        type=float,
        default=900.0,
        help="Band-pass low cutoff in Hz. Helps reject fan noise and room rumble.",
    )
    parser.add_argument(
        "--highcut",
        type=float,
        default=10500.0,
        help="Band-pass high cutoff in Hz.",
    )
    parser.add_argument(
        "--filter-order",
        type=int,
        default=6,
        help="Butterworth filter order.",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=0.15,
        help="Minimum reflector distance in meters for echo search.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=1.2,
        help="Maximum reflector distance in meters for echo search.",
    )
    parser.add_argument(
        "--min-echo-ratio",
        type=float,
        default=0.18,
        help="Minimum echo/direct peak ratio before reporting a distance.",
    )
    parser.add_argument(
        "--min-echo-snr",
        type=float,
        default=5.0,
        help="Minimum echo/noise ratio before reporting a distance.",
    )
    parser.add_argument(
        "--min-burst-snr",
        type=float,
        default=25.0,
        help="Minimum direct-burst/noise ratio before trusting that a chirp was detected.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Median filter length applied to the printed distance.",
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


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    # scale waveform to unit peak
    peak = np.max(np.abs(audio))
    if peak < 1e-12:
        return audio.astype(np.float32, copy=False)
    return (audio / peak).astype(np.float32, copy=False)


def to_mono_float(audio: np.ndarray) -> np.ndarray:
    # convert to mono float and remove dc offset
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio[:, 0]

    if np.issubdtype(audio.dtype, np.integer):
        scale = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / float(scale)
    else:
        audio = audio.astype(np.float32, copy=False)

    audio = audio - np.mean(audio)
    return audio


def resolve_device(device: str | None) -> int | str | None:
    # parse device as index when possible
    if device is None:
        return None
    try:
        return int(device)
    except ValueError:
        return device


def load_reference(reference_path: str) -> tuple[int, np.ndarray]:
    # load and normalize the preamble reference
    fs, preamble = wavfile.read(reference_path)
    preamble = normalize_audio(to_mono_float(preamble))
    return fs, preamble


def build_bandpass(fs: int, lowcut: float, highcut: float, order: int) -> np.ndarray:
    # design stable bandpass filter in sos form
    nyquist = fs / 2.0
    clipped_low = max(20.0, lowcut)
    clipped_high = min(highcut, nyquist * 0.95)
    if clipped_low >= clipped_high:
        raise ValueError(
            f"Invalid band-pass range: lowcut={lowcut} Hz, highcut={highcut} Hz, fs={fs} Hz"
        )
    return signal.butter(order, [clipped_low, clipped_high], btype="bandpass", fs=fs, output="sos")


def apply_bandpass(audio: np.ndarray, sos: np.ndarray) -> np.ndarray:
    # apply zero phase filter when possible
    if len(audio) < 64:
        return signal.sosfilt(sos, audio).astype(np.float32, copy=False)
    try:
        filtered = signal.sosfiltfilt(sos, audio)
    except ValueError:
        filtered = signal.sosfilt(sos, audio)
    return filtered.astype(np.float32, copy=False)


def estimate_distance(
    audio_window: np.ndarray,
    filtered_preamble: np.ndarray,
    sos: np.ndarray,
    fs: int,
    min_distance_m: float,
    max_distance_m: float,
    min_echo_ratio: float,
    min_echo_snr: float,
    min_burst_snr: float,
) -> EchoEstimate | None:
    # estimate direct and echo peaks from matched filter cir
    window = to_mono_float(audio_window)
    if np.max(np.abs(window)) < 1e-4:
        return None

    filtered_window = normalize_audio(apply_bandpass(window, sos))
    cir = np.abs(signal.fftconvolve(filtered_window, filtered_preamble[::-1], mode="same"))

    margin = len(filtered_preamble) // 2
    valid_start = margin
    valid_stop = len(cir) - margin
    if valid_stop <= valid_start:
        return None

    burst_noise_floor = float(np.median(cir[valid_start:valid_stop])) + 1e-12
    direct_idx = int(np.argmax(cir[valid_start:valid_stop])) + valid_start
    direct_peak = float(cir[direct_idx]) + 1e-12
    burst_snr = direct_peak / burst_noise_floor
    if burst_snr < min_burst_snr:
        return None

    min_gap = max(1, int(round((2.0 * min_distance_m) * fs / SPEED_OF_SOUND_M_S)))
    max_gap = max(min_gap + 1, int(round((2.0 * max_distance_m) * fs / SPEED_OF_SOUND_M_S)))

    search_start = direct_idx + min_gap
    search_stop = min(len(cir), direct_idx + max_gap)
    if search_stop <= search_start + 3:
        return None

    search_region = cir[search_start:search_stop]
    noise_floor = float(np.median(search_region)) + 1e-12
    min_height = max(noise_floor * 2.5, float(np.max(search_region)) * 0.35)
    min_prominence = max(noise_floor * 1.5, float(np.max(search_region)) * 0.10)

    peak_indices, properties = signal.find_peaks(
        search_region,
        height=min_height,
        prominence=min_prominence,
        distance=max(6, min_gap // 4),
    )

    if len(peak_indices) == 0:
        echo_idx = int(np.argmax(search_region)) + search_start
    else:
        peak_heights = properties["peak_heights"]
        peak_scores = properties["prominences"] + 0.35 * peak_heights
        echo_idx = int(peak_indices[int(np.argmax(peak_scores))]) + search_start

    echo_peak = float(cir[echo_idx])
    echo_ratio = echo_peak / direct_peak
    echo_snr = echo_peak / noise_floor
    if echo_ratio < min_echo_ratio or echo_snr < min_echo_snr:
        return None

    gap_samples = echo_idx - direct_idx
    distance_m = SPEED_OF_SOUND_M_S * gap_samples / fs / 2.0
    return EchoEstimate(
        distance_m=distance_m,
        direct_idx=direct_idx,
        echo_idx=echo_idx,
        direct_peak=direct_peak,
        echo_peak=echo_peak,
        echo_ratio=echo_ratio,
        echo_snr=echo_snr,
        burst_snr=burst_snr,
        cir=cir,
    )


def create_plot() -> tuple[plt.Figure, plt.Axes, object, tuple[object, object]]:
    # initialize live cir plot objects
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    line, = ax.plot([], [], linewidth=2)
    direct_marker = ax.axvline(0.0, color="tab:green", linestyle="--", linewidth=1.5, label="Direct")
    echo_marker = ax.axvline(0.0, color="tab:red", linestyle="--", linewidth=1.5, label="Echo")
    ax.set_title("Estimated CIR Around the Current Burst")
    ax.set_xlabel("Relative reflector distance from direct path (m)")
    ax.set_ylabel("Matched-filter magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, ax, line, (direct_marker, echo_marker)


def update_plot(
    ax: plt.Axes,
    line,
    markers,
    estimate: EchoEstimate,
    fs: int,
    max_distance_m: float,
    smoothed_distance_m: float,
) -> None:
    # refresh plot with latest cir and markers
    left_guard = max(80, int(0.02 * fs))
    right_guard = max(160, int(round((2.0 * max_distance_m) * fs / SPEED_OF_SOUND_M_S)) + 80)
    start = max(0, estimate.direct_idx - left_guard)
    stop = min(len(estimate.cir), estimate.direct_idx + right_guard)
    local_cir = estimate.cir[start:stop]
    axis_m = (np.arange(start, stop) - estimate.direct_idx) * SPEED_OF_SOUND_M_S / fs / 2.0

    line.set_data(axis_m, local_cir)
    markers[0].set_xdata([0.0, 0.0])
    markers[1].set_xdata([estimate.distance_m, estimate.distance_m])
    ax.set_xlim(float(axis_m[0]), float(axis_m[-1]))
    ax.set_ylim(0.0, float(np.max(local_cir)) * 1.1)
    ax.set_title(
        "Estimated CIR Around the Current Burst "
        f"(instant={estimate.distance_m:.3f} m, smooth={smoothed_distance_m:.3f} m)"
    )
    ax.figure.canvas.draw_idle()
    plt.pause(0.001)


def save_snapshot(
    output_path: str,
    estimate: EchoEstimate | None,
    fs: int,
    max_distance_m: float,
) -> None:
    # save final cir snapshot image
    if estimate is None:
        return

    fig, ax = plt.subplots(figsize=(10, 4.5))
    left_guard = max(80, int(0.02 * fs))
    right_guard = max(160, int(round((2.0 * max_distance_m) * fs / SPEED_OF_SOUND_M_S)) + 80)
    start = max(0, estimate.direct_idx - left_guard)
    stop = min(len(estimate.cir), estimate.direct_idx + right_guard)
    local_cir = estimate.cir[start:stop]
    axis_m = (np.arange(start, stop) - estimate.direct_idx) * SPEED_OF_SOUND_M_S / fs / 2.0

    ax.plot(axis_m, local_cir, linewidth=2)
    ax.axvline(0.0, color="tab:green", linestyle="--", linewidth=1.5, label="Direct")
    ax.axvline(estimate.distance_m, color="tab:red", linestyle="--", linewidth=1.5, label="Echo")
    ax.set_title(f"CIR Snapshot (estimated distance = {estimate.distance_m:.3f} m)")
    ax.set_xlabel("Relative reflector distance from direct path (m)")
    ax.set_ylabel("Matched-filter magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_file_mode(
    audio_path: str,
    filtered_preamble: np.ndarray,
    sos: np.ndarray,
    fs: int,
    args: argparse.Namespace,
) -> EchoEstimate | None:
    # scan wav with sliding windows and keep valid estimates
    file_fs, audio = wavfile.read(audio_path)
    if file_fs != fs:
        raise ValueError(
            f"Input file sample rate is {file_fs} Hz, but preamble/sample rate is {fs} Hz."
        )

    audio = to_mono_float(audio)
    window_samples = int(round(args.window_seconds * fs))
    step_samples = max(1, int(round(args.step_seconds * fs)))
    smoothing = deque(maxlen=max(1, args.smooth))
    accepted_estimates: list[tuple[float, EchoEstimate]] = []

    for start in range(0, max(1, len(audio) - window_samples + 1), step_samples):
        window = audio[start : start + window_samples]
        if len(window) < window_samples:
            window = np.pad(window, (0, window_samples - len(window)))

        estimate = estimate_distance(
            window,
            filtered_preamble,
            sos,
            fs,
            args.min_distance,
            args.max_distance,
            args.min_echo_ratio,
            args.min_echo_snr,
            args.min_burst_snr,
        )
        if estimate is None:
            continue

        smoothing.append(estimate.distance_m)
        smooth_distance = float(np.median(np.asarray(smoothing)))
        print(
            f"t={start / fs:6.2f}s  "
            f"instant={estimate.distance_m:0.3f} m  "
            f"smooth={smooth_distance:0.3f} m  "
            f"echo/direct={estimate.echo_ratio:0.2f}  "
            f"echo/noise={estimate.echo_snr:0.1f}  "
            f"burst/noise={estimate.burst_snr:0.1f}"
        )
        accepted_estimates.append((start / fs, estimate))

    if not accepted_estimates:
        print("No clear reflector echo was detected in the WAV file.")
        return None

    distances = np.asarray([estimate.distance_m for _, estimate in accepted_estimates], dtype=np.float32)
    median_distance = float(np.median(distances))
    representative_time, representative_estimate = min(
        accepted_estimates,
        key=lambda item: (
            abs(item[1].distance_m - median_distance),
            -item[1].echo_snr,
        ),
    )
    print(
        f"Median distance from file: {median_distance:.3f} m "
        f"(representative window at t={representative_time:.2f}s, "
        f"echo/direct={representative_estimate.echo_ratio:.2f}, "
        f"echo/noise={representative_estimate.echo_snr:.1f}, "
        f"burst/noise={representative_estimate.burst_snr:.1f})"
    )
    return representative_estimate


def run_live_mode(
    filtered_preamble: np.ndarray,
    sos: np.ndarray,
    fs: int,
    args: argparse.Namespace,
) -> EchoEstimate | None:
    # stream mic audio and print rolling distance estimates
    if sd is None:
        raise RuntimeError(
            "Live mode requires sounddevice. Install it with: pip install sounddevice"
        )

    device = resolve_device(args.device)
    window_samples = int(round(args.window_seconds * fs))
    chunk_samples = max(1, int(round(args.chunk_seconds * fs)))
    rolling_audio = np.zeros(window_samples, dtype=np.float32)
    collected = 0
    last_estimate = None
    smoothing = deque(maxlen=max(1, args.smooth))

    plot_state = None
    if not args.no_plot:
        plot_state = create_plot()

    print(f"Listening at {fs} Hz. Press Ctrl+C to stop.")
    if device is not None:
        print(f"Using input device: {device}")

    start_time = time.time()
    with sd.InputStream(
        samplerate=fs,
        blocksize=chunk_samples,
        channels=1,
        dtype="float32",
        device=device,
        latency="low",
    ) as stream:
        while True:
            if args.duration is not None and (time.time() - start_time) >= args.duration:
                break

            chunk, overflowed = stream.read(chunk_samples)
            samples = np.asarray(chunk[:, 0], dtype=np.float32)
            rolling_audio = np.roll(rolling_audio, -len(samples))
            rolling_audio[-len(samples) :] = samples
            collected = min(window_samples, collected + len(samples))

            if overflowed:
                print("\nWarning: input overflow detected. Try a larger chunk size.")

            if collected < window_samples:
                continue

            estimate = estimate_distance(
                rolling_audio,
                filtered_preamble,
                sos,
                fs,
                args.min_distance,
                args.max_distance,
                args.min_echo_ratio,
                args.min_echo_snr,
                args.min_burst_snr,
            )
            if estimate is None:
                print("\rListening... no clean echo yet.                                ", end="", flush=True)
                continue

            last_estimate = estimate
            smoothing.append(estimate.distance_m)
            smooth_distance = float(np.median(np.asarray(smoothing)))
            print(
                "\r"
                f"Distance={smooth_distance:0.3f} m  "
                f"instant={estimate.distance_m:0.3f} m  "
                f"echo/direct={estimate.echo_ratio:0.2f}  "
                f"echo/noise={estimate.echo_snr:0.1f}  "
                f"burst/noise={estimate.burst_snr:0.1f}      ",
                end="",
                flush=True,
            )

            if plot_state is not None:
                _, ax, line, markers = plot_state
                update_plot(ax, line, markers, estimate, fs, args.max_distance, smooth_distance)

    print()
    return last_estimate


def print_device_list() -> None:
    # print available audio input and output devices
    if sd is None:
        raise RuntimeError("sounddevice is not installed, so devices cannot be listed.")
    print(sd.query_devices())


def main() -> None:
    # wire setup then run file mode or live mode
    args = parse_args()

    if args.list_devices:
        print_device_list()
        return

    fs, preamble = load_reference(args.preamble)
    if args.samplerate is not None and args.samplerate != fs:
        raise ValueError(
            "This script assumes the incoming audio and preamble use the same sample rate. "
            f"preamble={fs} Hz, requested={args.samplerate} Hz"
        )

    sos = build_bandpass(fs, args.lowcut, args.highcut, args.filter_order)
    filtered_preamble = normalize_audio(apply_bandpass(preamble, sos))
    best_estimate = None

    try:
        if args.input_file:
            best_estimate = run_file_mode(args.input_file, filtered_preamble, sos, fs, args)
        else:
            best_estimate = run_live_mode(filtered_preamble, sos, fs, args)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        save_snapshot(args.save_plot, best_estimate, fs, args.max_distance)
        if best_estimate is not None:
            print(f"Saved CIR snapshot to {args.save_plot}")


if __name__ == "__main__":
    # script entry point
    main()

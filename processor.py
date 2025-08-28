import os
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import cv2
from tqdm import tqdm

import librosa
import noisereduce as nr
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.audio.AudioClip import AudioArrayClip


SecondsInterval = Tuple[float, float]


@dataclass
class ProcessingParams:
    enable_denoise: bool = True
    enable_pause_removal: bool = True
    enable_blur_removal: bool = True

    # Audio / silence detection
    rms_threshold_db: float = -35.0  # below this is considered silence
    min_silence_len_s: float = 0.6   # seconds

    # Blur detection
    blur_variance_threshold: float = 60.0
    min_blur_len_s: float = 0.3

    # Segment merging and margins
    merge_distance_s: float = 0.1
    cut_margin_s: float = 0.03


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def denoise_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Denoise audio per channel using spectral gating.

    Expects shape (n_samples,) for mono or (n_samples, n_channels) for multi.
    Returns same-shape array.
    """
    if audio.ndim == 1:
        return nr.reduce_noise(y=audio.astype(np.float32), sr=sample_rate)

    # Stereo or multi-channel: process channel-wise
    denoised_channels: List[np.ndarray] = []
    for channel_idx in range(audio.shape[1]):
        channel = audio[:, channel_idx].astype(np.float32)
        denoised = nr.reduce_noise(y=channel, sr=sample_rate)
        denoised_channels.append(denoised)
    return np.stack(denoised_channels, axis=1)


def detect_silence_intervals(
    audio: np.ndarray,
    sample_rate: int,
    rms_threshold_db: float,
    min_silence_len_s: float,
) -> List[SecondsInterval]:
    """Detect silence intervals based on RMS energy in dB.

    Returns list of (start_s, end_s).
    """
    if audio.ndim > 1:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio

    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=mono.astype(np.float32), frame_length=frame_length, hop_length=hop_length)[0]
    eps = 1e-9
    rms_db = 20.0 * np.log10(np.maximum(rms, eps))

    times = librosa.frames_to_time(np.arange(len(rms)), sr=sample_rate, hop_length=hop_length)

    is_silence = rms_db < rms_threshold_db

    intervals: List[SecondsInterval] = []
    start_idx: Optional[int] = None
    for i, silent in enumerate(is_silence):
        if silent and start_idx is None:
            start_idx = i
        elif not silent and start_idx is not None:
            start_t = float(times[start_idx])
            end_t = float(times[i])
            if end_t - start_t >= min_silence_len_s:
                intervals.append((start_t, end_t))
            start_idx = None
    # tail
    if start_idx is not None:
        start_t = float(times[start_idx])
        end_t = float(times[-1])
        if end_t - start_t >= min_silence_len_s:
            intervals.append((start_t, end_t))

    return intervals


def detect_blurry_intervals(
    video_path: str,
    variance_threshold: float,
    min_len_s: float,
) -> List[SecondsInterval]:
    """Scan frames and mark intervals where frames are blurry using Laplacian variance.
    Returns (start_s, end_s) intervals.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    blurry_intervals: List[SecondsInterval] = []
    start_frame_idx: Optional[int] = None

    frame_idx = 0
    success, frame = cap.read()
    while success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = variance < variance_threshold

        if is_blurry and start_frame_idx is None:
            start_frame_idx = frame_idx
        elif not is_blurry and start_frame_idx is not None:
            start_t = start_frame_idx / fps
            end_t = frame_idx / fps
            if (end_t - start_t) >= min_len_s:
                blurry_intervals.append((start_t, end_t))
            start_frame_idx = None

        frame_idx += 1
        success, frame = cap.read()

    # tail
    if start_frame_idx is not None:
        start_t = start_frame_idx / fps
        end_t = frame_idx / fps
        if (end_t - start_t) >= min_len_s:
            blurry_intervals.append((start_t, end_t))

    cap.release()
    return blurry_intervals


def merge_intervals(
    intervals: Sequence[SecondsInterval],
    merge_distance_s: float,
    clip_start_s: float,
    clip_end_s: float,
) -> List[SecondsInterval]:
    if not intervals:
        return []
    sorted_intervals = sorted(((max(a, clip_start_s), min(b, clip_end_s)) for a, b in intervals), key=lambda x: x[0])
    merged: List[SecondsInterval] = []
    cur_start, cur_end = sorted_intervals[0]
    for a, b in sorted_intervals[1:]:
        if a <= cur_end + merge_distance_s:
            cur_end = max(cur_end, b)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = a, b
    merged.append((cur_start, cur_end))
    return merged


def expand_intervals(intervals: Sequence[SecondsInterval], margin_s: float, bounds: SecondsInterval) -> List[SecondsInterval]:
    if margin_s <= 0:
        return list(intervals)
    start_bound, end_bound = bounds
    return [(max(start_bound, a - margin_s), min(end_bound, b + margin_s)) for a, b in intervals]


def invert_intervals(
    remove_intervals: Sequence[SecondsInterval], start_s: float, end_s: float
) -> List[SecondsInterval]:
    if not remove_intervals:
        return [(start_s, end_s)]
    keep: List[SecondsInterval] = []
    cur = start_s
    for a, b in remove_intervals:
        if a > cur:
            keep.append((cur, a))
        cur = max(cur, b)
    if cur < end_s:
        keep.append((cur, end_s))
    return keep


def cut_audio_by_intervals(audio: np.ndarray, sample_rate: int, keep_intervals: Sequence[SecondsInterval]) -> np.ndarray:
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    parts: List[np.ndarray] = []
    for a, b in keep_intervals:
        start_idx = max(0, int(round(a * sample_rate)))
        end_idx = min(len(audio), int(round(b * sample_rate)))
        if end_idx <= start_idx:
            continue
        if channels == 1:
            segment = audio[start_idx:end_idx]
        else:
            segment = audio[start_idx:end_idx, :]
        parts.append(segment)
    if not parts:
        return np.zeros((0, channels), dtype=audio.dtype) if channels > 1 else np.zeros((0,), dtype=audio.dtype)
    return np.concatenate(parts, axis=0)


def cut_and_concatenate_video(
    video_path: str,
    keep_intervals: Sequence[SecondsInterval],
    denoised_audio: Optional[np.ndarray],
    audio_sr: Optional[int],
    output_path: str,
) -> str:
    clip = VideoFileClip(video_path)
    subclips = []
    for a, b in keep_intervals:
        if b - a <= 0.02:
            continue
        sub = clip.subclip(a, b)
        subclips.append(sub)
    if not subclips:
        clip.close()
        raise RuntimeError("All segments were removed; no output produced.")

    out_clip = concatenate_videoclips(subclips, method="compose")

    # Replace audio if provided
    if denoised_audio is not None and audio_sr is not None:
        if denoised_audio.ndim == 1:
            arr = denoised_audio.astype(np.float32)
        else:
            arr = denoised_audio.astype(np.float32)
        audio_clip = AudioArrayClip(arr, fps=audio_sr)
        out_clip = out_clip.set_audio(audio_clip)

    ensure_dir(os.path.dirname(output_path))
    out_clip.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        audio_bitrate="192k",
        threads=os.cpu_count() or 4,
        verbose=False,
        logger=None,
    )

    out_clip.close()
    clip.close()
    for sc in subclips:
        sc.close()
    return output_path


def process_video(
    video_path: str,
    output_dir: str,
    params: ProcessingParams,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Tuple[str, dict]:
    ensure_dir(output_dir)
    if progress_cb:
        progress_cb(0.05, "Loading video")

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    temp_audio_sr = 44100

    with VideoFileClip(video_path) as vclip:
        duration = float(vclip.duration)
        aud = vclip.audio
        if aud is None:
            # synthesize silent audio to keep pipeline consistent
            raw_audio = np.zeros((int(duration * temp_audio_sr),), dtype=np.float32)
            sr = temp_audio_sr
        else:
            sr = int(aud.fps)
            raw_audio = aud.to_soundarray(fps=sr)

    # Ensure float32
    if raw_audio.dtype != np.float32:
        raw_audio = raw_audio.astype(np.float32)

    if params.enable_denoise:
        if progress_cb:
            progress_cb(0.15, "Denoising audio")
        denoised = denoise_audio(raw_audio, sr)
    else:
        denoised = raw_audio

    if params.enable_pause_removal:
        if progress_cb:
            progress_cb(0.35, "Detecting pauses/silence")
        silence_intervals = detect_silence_intervals(
            denoised, sr, params.rms_threshold_db, params.min_silence_len_s
        )
    else:
        silence_intervals = []

    if params.enable_blur_removal:
        if progress_cb:
            progress_cb(0.55, "Scanning for blurry scenes")
        blur_intervals = detect_blurry_intervals(
            video_path, params.blur_variance_threshold, params.min_blur_len_s
        )
    else:
        blur_intervals = []

    if progress_cb:
        progress_cb(0.7, "Merging intervals")
    remove_intervals = silence_intervals + blur_intervals
    remove_intervals = expand_intervals(remove_intervals, params.cut_margin_s, (0.0, duration))
    remove_intervals = merge_intervals(remove_intervals, params.merge_distance_s, 0.0, duration)

    keep_intervals = invert_intervals(remove_intervals, 0.0, duration)

    if progress_cb:
        progress_cb(0.78, "Cutting audio")
    final_audio = cut_audio_by_intervals(denoised, sr, keep_intervals)

    if progress_cb:
        progress_cb(0.9, "Rendering video")
    out_path = os.path.join(output_dir, f"{base_name}_cleaned.mp4")
    result_path = cut_and_concatenate_video(
        video_path, keep_intervals, final_audio, sr, out_path
    )

    if progress_cb:
        progress_cb(1.0, "Done")

    metrics = {
        "duration_in": duration,
        "num_remove_segments": len(remove_intervals),
        "num_keep_segments": len(keep_intervals),
    }
    return result_path, metrics


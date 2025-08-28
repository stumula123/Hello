import math
import struct
from typing import List, Tuple, Optional

import numpy as np
import webrtcvad
import noisereduce as nr


def float_to_pcm16_bytes(audio_float: np.ndarray) -> bytes:
	"""Convert float32 [-1, 1] to PCM16 bytes."""
	clipped = np.clip(audio_float, -1.0, 1.0)
	int16 = (clipped * 32767.0).astype(np.int16)
	return int16.tobytes()


def generate_frames(pcm16_bytes: bytes, sample_rate: int, frame_ms: int) -> List[bytes]:
	"""Yield fixed-size frames of PCM16 bytes corresponding to frame_ms."""
	frame_bytes = int(sample_rate * (frame_ms / 1000.0)) * 2  # 2 bytes per sample mono
	frames = []
	for start in range(0, len(pcm16_bytes), frame_bytes):
		end = start + frame_bytes
		if end > len(pcm16_bytes):
			break
		frames.append(pcm16_bytes[start:end])
	return frames


def merge_voiced_regions(
	voiced_flags: List[bool],
	frame_ms: int,
	pre_pad_ms: int,
	post_pad_ms: int,
	min_speech_ms: int,
) -> List[Tuple[float, float]]:
	"""Merge contiguous voiced frames, apply padding, and return (start_sec, end_sec)."""
	segments: List[Tuple[int, int]] = []
	in_segment = False
	start_idx = 0
	for idx, is_voiced in enumerate(voiced_flags):
		if is_voiced and not in_segment:
			in_segment = True
			start_idx = idx
		elif not is_voiced and in_segment:
			in_segment = False
			segments.append((start_idx, idx))
	# Close trailing
	if in_segment:
		segments.append((start_idx, len(voiced_flags)))

	# Apply padding and convert to seconds
	merged: List[Tuple[float, float]] = []
	pad_pre_frames = int(pre_pad_ms / frame_ms)
	pad_post_frames = int(post_pad_ms / frame_ms)
	min_frames = max(1, int(math.ceil(min_speech_ms / frame_ms)))
	for start_f, end_f in segments:
		start_f = max(0, start_f - pad_pre_frames)
		end_f = end_f + pad_post_frames
		if end_f - start_f >= min_frames:
			merged.append((start_f * frame_ms / 1000.0, end_f * frame_ms / 1000.0))

	# Merge overlaps
	if not merged:
		return []
	merged.sort()
	result: List[Tuple[float, float]] = []
	cur_s, cur_e = merged[0]
	for s, e in merged[1:]:
		if s <= cur_e:
			cur_e = max(cur_e, e)
		else:
			result.append((cur_s, cur_e))
			cur_s, cur_e = s, e
	result.append((cur_s, cur_e))
	return result


def compute_vad_segments(
	audio_float: np.ndarray,
	sample_rate: int,
	aggressiveness: int = 2,
	frame_ms: int = 30,
	pre_pad_ms: int = 120,
	post_pad_ms: int = 120,
	min_speech_ms: int = 150,
) -> List[Tuple[float, float]]:
	"""
	Returns list of (start_sec, end_sec) for speech segments.
	"""
	if sample_rate not in (8000, 16000, 32000, 48000):
		raise ValueError("WebRTC VAD supports 8k/16k/32k/48k sample rates only")
	if frame_ms not in (10, 20, 30):
		raise ValueError("WebRTC VAD supports 10, 20, or 30 ms frames")
	vad = webrtcvad.Vad(aggressiveness)
	pcm_bytes = float_to_pcm16_bytes(audio_float)
	frames = generate_frames(pcm_bytes, sample_rate, frame_ms)
	voiced = [vad.is_speech(frame, sample_rate) for frame in frames]
	segments = merge_voiced_regions(voiced, frame_ms, pre_pad_ms, post_pad_ms, min_speech_ms)
	return segments


def extract_noise_clip(
	audio_float: np.ndarray,
	sample_rate: int,
	speech_segments: List[Tuple[float, float]],
	fallback_seconds: float = 0.5,
) -> np.ndarray:
	"""Extract a noise-only clip using the longest non-speech interval; fallback to start of file."""
	if not speech_segments:
		# No speech found; use whole clip as noise baseline
		return audio_float[: int(fallback_seconds * sample_rate)]
	# Build complement intervals (non-speech)
	speech_segments_sorted = sorted(speech_segments)
	non_speech: List[Tuple[float, float]] = []
	cursor = 0.0
	end_total = len(audio_float) / sample_rate
	for s, e in speech_segments_sorted:
		if s > cursor:
			non_speech.append((cursor, s))
		cursor = max(cursor, e)
	if cursor < end_total:
		non_speech.append((cursor, end_total))

	if not non_speech:
		return audio_float[: int(fallback_seconds * sample_rate)]
	# Pick the longest non-speech interval
	lengths = [e - s for s, e in non_speech]
	best_idx = int(np.argmax(lengths))
	start, end = non_speech[best_idx]
	start_i = int(start * sample_rate)
	end_i = int(end * sample_rate)
	clip = audio_float[start_i:end_i]
	if clip.size < int(0.1 * sample_rate):
		return audio_float[: int(fallback_seconds * sample_rate)]
	return clip


def denoise_audio(audio_float: np.ndarray, sample_rate: int, noise_clip: Optional[np.ndarray]) -> np.ndarray:
	"""Apply spectral noise reduction using noisereduce."""
	if noise_clip is None or noise_clip.size < int(0.05 * sample_rate):
		# Stationary reduction if no noise reference
		return nr.reduce_noise(y=audio_float, sr=sample_rate, stationary=True)
	return nr.reduce_noise(y=audio_float, sr=sample_rate, y_noise=noise_clip, stationary=False)


def compute_rms(audio_float: np.ndarray) -> float:
	"""Root mean square of audio."""
	if audio_float.size == 0:
		return 1e-9
	return float(np.sqrt(np.mean(np.square(audio_float.astype(np.float64))) + 1e-12))


def select_segments_by_snr(
	speech_segments: List[Tuple[float, float]],
	audio_float: np.ndarray,
	sample_rate: int,
	noise_rms: float,
	snr_threshold_db: float,
) -> List[Tuple[float, float]]:
	"""Keep segments where segment RMS exceeds noise RMS by threshold in dB."""
	if not speech_segments:
		return []
	kept: List[Tuple[float, float]] = []
	for s, e in speech_segments:
		start_i = int(s * sample_rate)
		end_i = int(e * sample_rate)
		seg = audio_float[start_i:end_i]
		seg_rms = compute_rms(seg)
		snr_db = 20.0 * math.log10(max(seg_rms, 1e-9) / max(noise_rms, 1e-9))
		if snr_db >= snr_threshold_db:
			kept.append((s, e))
	return kept
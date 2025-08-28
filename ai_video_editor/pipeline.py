import os
import tempfile
from typing import List, Tuple

import numpy as np
import soundfile as sf
import librosa
import ffmpeg
from moviepy import VideoFileClip, concatenate_videoclips
from moviepy.audio.AudioClip import AudioArrayClip

from .audio_utils import (
	compute_vad_segments,
	extract_noise_clip,
	denoise_audio,
	compute_rms,
	select_segments_by_snr,
)


def _extract_audio_to_wav(input_video: str, out_wav: str, sample_rate: int = 16000) -> None:
	(
		ffmpeg
		.input(input_video)
		.output(
			out_wav,
			vn=None,
			ac=1,
			ar=sample_rate,
			acodec='pcm_s16le',
		)
		.overwrite_output()
		.run(quiet=True)
	)


def _load_audio_float_mono(path: str, expected_sr: int) -> np.ndarray:
	audio, sr = sf.read(path, dtype='float32')
	if audio.ndim == 2:
		audio = np.mean(audio, axis=1)
	if sr != expected_sr:
		audio = librosa.resample(audio, orig_sr=sr, target_sr=expected_sr)
	return audio.astype(np.float32)


def _build_kept_audio(
	audio_float: np.ndarray,
	sample_rate: int,
	segments: List[Tuple[float, float]],
) -> np.ndarray:
	parts: List[np.ndarray] = []
	for s, e in segments:
		start_i = int(max(0.0, s) * sample_rate)
		end_i = int(max(s, e) * sample_rate)
		parts.append(audio_float[start_i:end_i])
	if not parts:
		return np.zeros((0,), dtype=np.float32)
	return np.concatenate(parts, axis=0)


def process_video(
	input_path: str,
	output_path: str,
	vad_aggressiveness: int = 2,
	snr_threshold_db: float = 5.0,
	pre_pad_ms: int = 120,
	post_pad_ms: int = 120,
	frame_ms: int = 30,
	min_speech_ms: int = 150,
	output_audio_fps: int = 44100,
) -> None:
	"""
	Process a video: denoise audio, remove pauses and low-SNR segments, keep A/V in sync.
	"""
	if not os.path.exists(input_path):
		raise FileNotFoundError(f"Input not found: {input_path}")
	os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

	with tempfile.TemporaryDirectory() as tmpdir:
		wav_path = os.path.join(tmpdir, 'audio_16k_mono.wav')
		_extract_audio_to_wav(input_path, wav_path, sample_rate=16000)

		audio_16k = _load_audio_float_mono(wav_path, expected_sr=16000)

		segments_vad = compute_vad_segments(
			audio_16k,
			16000,
			aggressiveness=vad_aggressiveness,
			frame_ms=frame_ms,
			pre_pad_ms=pre_pad_ms,
			post_pad_ms=post_pad_ms,
			min_speech_ms=min_speech_ms,
		)
		if not segments_vad:
			raise RuntimeError("No speech detected. Try lowering VAD aggressiveness or check input.")

		noise_clip = extract_noise_clip(audio_16k, 16000, segments_vad)
		audio_denoised = denoise_audio(audio_16k, 16000, noise_clip)

		noise_rms = compute_rms(noise_clip)
		segments_kept = select_segments_by_snr(
			speech_segments=segments_vad,
			audio_float=audio_denoised,
			sample_rate=16000,
			noise_rms=noise_rms,
			snr_threshold_db=snr_threshold_db,
		)
		if not segments_kept:
			raise RuntimeError("All segments dropped by SNR threshold. Try lowering --snr-threshold-db.")

		# Build denoised, trimmed audio
		kept_audio_16k = _build_kept_audio(audio_denoised, 16000, segments_kept)
		if kept_audio_16k.size == 0:
			raise RuntimeError("Kept audio is empty after trimming.")

		# Resample for video output
		kept_audio_out = librosa.resample(kept_audio_16k, orig_sr=16000, target_sr=output_audio_fps)
		kept_audio_out = kept_audio_out.astype(np.float32)
		# Shape for AudioArrayClip: (n_samples, n_channels)
		audio_arr = kept_audio_out.reshape(-1, 1)

		# Cut video to the exact same segments
		video = VideoFileClip(input_path)
		clips = [video.subclip(max(0.0, s), max(s, e)) for s, e in segments_kept]
		final_video = concatenate_videoclips(clips, method='compose')

		# Duration alignment safety: trim/pad audio to match final_video duration
		target_duration = final_video.duration
		current_duration = audio_arr.shape[0] / float(output_audio_fps)
		if abs(current_duration - target_duration) > 1e-3:
			# Trim or pad with small silence to match
			n_target = int(round(target_duration * output_audio_fps))
			n_current = audio_arr.shape[0]
			if n_current >= n_target:
				audio_arr = audio_arr[:n_target, :]
			else:
				pad = np.zeros((n_target - n_current, 1), dtype=audio_arr.dtype)
				audio_arr = np.concatenate([audio_arr, pad], axis=0)

		audio_clip = AudioArrayClip(audio_arr, fps=output_audio_fps)
		final = final_video.set_audio(audio_clip)

		# Write output
		out_temp_audio = os.path.join(tmpdir, 'temp_audio.m4a')
		final.write_videofile(
			output_path,
			codec='libx264',
			audio_codec='aac',
			audio_fps=output_audio_fps,
			temp_audiofile=out_temp_audio,
			remove_temp=True,
			threads=0,
			preset='medium',
			verbose=False,
			logger=None,
		)

		# Cleanup
		final.close()
		final_video.close()
		for c in clips:
			c.close()
		video.close()
import argparse
from .pipeline import process_video


def main() -> None:
	parser = argparse.ArgumentParser(description="AI video editor: denoise + trim pauses/disturbances")
	parser.add_argument("--input", required=True, help="Path to input video file")
	parser.add_argument("--output", required=True, help="Path to output video file")
	parser.add_argument("--vad-aggressiveness", type=int, default=2, choices=[0,1,2,3], help="VAD aggressiveness (0-3)")
	parser.add_argument("--snr-threshold-db", type=float, default=5.0, help="Drop segments with SNR below this dB")
	parser.add_argument("--pre-pad-ms", type=int, default=120, help="Padding before detected speech (ms)")
	parser.add_argument("--post-pad-ms", type=int, default=120, help="Padding after detected speech (ms)")
	parser.add_argument("--frame-ms", type=int, default=30, choices=[10,20,30], help="Frame size for VAD (ms)")
	parser.add_argument("--min-speech-ms", type=int, default=150, help="Minimum speech segment length (ms)")
	parser.add_argument("--output-audio-fps", type=int, default=44100, help="Audio sample rate for output video")
	args = parser.parse_args()

	process_video(
		input_path=args.input,
		output_path=args.output,
		vad_aggressiveness=args.vad_aggressiveness,
		snr_threshold_db=args.snr_threshold_db,
		pre_pad_ms=args.pre_pad_ms,
		post_pad_ms=args.post_pad_ms,
		frame_ms=args.frame_ms,
		min_speech_ms=args.min_speech_ms,
		output_audio_fps=args.output_audio_fps,
	)


if __name__ == "__main__":
	main()
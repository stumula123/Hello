### AI Video Editor (Noise Removal + Pause/Disturbance Trimming)

This project takes a noisy video with pauses and disturbed segments, denoises the audio, removes pauses and low-SNR (disturbed) segments, and outputs a synced, cleaner video.

### Features
- **Background noise reduction** using spectral noise reduction
- **Voice Activity Detection (VAD)** to remove pauses
- **Disturbed scene removal** by dropping segments with low SNR
- **A/V sync preserved** after trimming and denoising

### Requirements
- Python 3.9+
- ffmpeg installed and on PATH

Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Usage
```bash
python -m ai_video_editor.cli \
  --input /path/to/input.mp4 \
  --output /path/to/output_clean.mp4 \
  --vad-aggressiveness 2 \
  --snr-threshold-db 5 \
  --pre-pad-ms 120 \
  --post-pad-ms 120
```

- **input**: path to input video
- **output**: path to output video
- **vad-aggressiveness**: 0 (least aggressive) to 3 (most). Default 2
- **snr-threshold-db**: drop voiced segments below this SNR. Default 5 dB
- **pre/post-pad-ms**: padding around detected speech to avoid cutting too tightly

### How it works
1. Extracts mono 16kHz audio with ffmpeg
2. Runs WebRTC VAD to find speech frames, merges to segments, adds padding
3. Estimates noise from non-speech frames and applies spectral noise reduction
4. Computes per-segment SNR and drops segments below threshold
5. Cuts video to the kept segments and replaces audio with the denoised, aligned audio
6. Exports a single clean, synced video

### Notes
- For best results, ensure ffmpeg is installed: `ffmpeg -version`
- If there are no speech segments detected, the tool will exit with an error
- This pipeline re-encodes the output once (H.264 + AAC)

### License
MIT
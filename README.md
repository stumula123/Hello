## AI Video Cleaner (Noise Removal, Pause & Disturbed Scene Deletion)

This project provides a simple Streamlit web app (runs on localhost) to clean up a video:

- Remove audio background noise (spectral gating)
- Remove long pauses/silences in the audio
- Drop disturbed scenes (blurry frames)

The app outputs a processed MP4 video.

### Prerequisites

- Python 3.10+
- FFmpeg (required by MoviePy)

Install FFmpeg on Linux:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

### Setup

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open the browser at `http://localhost:8501`.

### Usage

1. Upload a video (MP4/MOV/MKV)
2. Choose which steps to enable: denoise, remove pauses, drop blurry scenes
3. Adjust thresholds as needed
4. Click Process to generate the cleaned video
5. Preview and download the output

### Notes

- Processing time depends on video length and CPU speed
- Large files may need more memory/time
- Blur detection uses Laplacian variance; adjust threshold if it removes too much/little
- Silence detection is based on RMS energy in dB; tweak threshold and min length

### Project Structure

```
app.py              # Streamlit UI
processor.py        # Video/audio processing pipeline
requirements.txt    # Python dependencies
```

# Hello
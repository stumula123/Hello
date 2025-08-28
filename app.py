import os
import time
import tempfile
from datetime import datetime

import streamlit as st

from processor import ProcessingParams, process_video


st.set_page_config(page_title="AI Video Cleaner", layout="centered")
st.title("AI Video Cleaner")
st.caption("Remove background noise, delete pauses, and drop blurry scenes")


def save_upload(uploaded_file) -> str:
    uploads_dir = os.path.join(os.getcwd(), "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    filename = f"{int(time.time())}_{uploaded_file.name}"
    dest = os.path.join(uploads_dir, filename)
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


with st.sidebar:
    st.header("Options")
    enable_denoise = st.checkbox("Denoise audio", value=True)
    enable_pause = st.checkbox("Remove pauses/silence", value=True)
    enable_blur = st.checkbox("Drop blurry scenes", value=True)

    st.subheader("Silence Detection")
    rms_db = st.slider("Silence threshold (dB)", min_value=-60, max_value=-10, value=-35)
    min_silence = st.slider("Min silence length (s)", min_value=0.2, max_value=2.0, value=0.6, step=0.1)

    st.subheader("Blur Detection")
    blur_var = st.slider("Laplacian variance threshold", min_value=20, max_value=300, value=60)
    min_blur_len = st.slider("Min blurry segment (s)", min_value=0.1, max_value=2.0, value=0.3, step=0.1)

    st.subheader("Cutting")
    merge_dist = st.slider("Merge nearby removals (s)", min_value=0.0, max_value=0.5, value=0.1, step=0.01)
    cut_margin = st.slider("Extra cut margin (s)", min_value=0.0, max_value=0.2, value=0.03, step=0.01)


uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "mkv", "m4v"])
process_btn = st.button("Process", type="primary", disabled=(uploaded is None))


if 'last_output' not in st.session_state:
    st.session_state['last_output'] = None
    st.session_state['last_metrics'] = None


if process_btn and uploaded is not None:
    video_path = save_upload(uploaded)
    st.write(f"Saved upload to `{video_path}`")

    params = ProcessingParams(
        enable_denoise=enable_denoise,
        enable_pause_removal=enable_pause,
        enable_blur_removal=enable_blur,
        rms_threshold_db=float(rms_db),
        min_silence_len_s=float(min_silence),
        blur_variance_threshold=float(blur_var),
        min_blur_len_s=float(min_blur_len),
        merge_distance_s=float(merge_dist),
        cut_margin_s=float(cut_margin),
    )

    progress = st.progress(0)
    status = st.empty()

    def cb(pct: float, msg: str):
        progress.progress(min(100, int(pct * 100)))
        status.info(msg)

    outputs_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    try:
        out_path, metrics = process_video(video_path, outputs_dir, params, progress_cb=cb)
        st.success("Processing complete!")
        st.session_state['last_output'] = out_path
        st.session_state['last_metrics'] = metrics
    except Exception as e:
        st.error(f"Processing failed: {e}")


if st.session_state['last_output']:
    st.subheader("Result")
    st.video(st.session_state['last_output'])
    with open(st.session_state['last_output'], 'rb') as f:
        st.download_button("Download processed video", f, file_name=os.path.basename(st.session_state['last_output']))
    if st.session_state['last_metrics']:
        st.json(st.session_state['last_metrics'])

st.caption("Powered by Streamlit, MoviePy, Librosa, OpenCV, and Noisereduce")


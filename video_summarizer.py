import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from moviepy.editor import VideoFileClip
import whisper

st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #141E30, #243B55);
        color: white;
    }
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: #FFD700;
        margin-bottom: 20px;
        text-shadow: 1px 1px 5px black;
    }
    .stButton>button {
        background-color: #FFD700 !important;
        color: black !important;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .box {
        background: rgba(255,255,255,0.1);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 30px rgba(0,0,0,0.3);
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("Please set GEMINI_API_KEY in your .env file")
    st.stop()

genai.configure(api_key=API_KEY)

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

model_whisper = load_whisper_model()

# ---------- Functions ----------
def extract_audio(video_file):
    """Extract audio from uploaded video and save as WAV."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        temp_video_path = temp_video.name

    clip = VideoFileClip(temp_video_path)
    audio_path = temp_video_path.replace(".mp4", ".wav")
    clip.audio.write_audiofile(audio_path)
    return audio_path

def transcribe_audio_whisper(audio_path):
    """Transcribe audio using OpenAI Whisper."""
    result = model_whisper.transcribe(audio_path)
    return result["text"]

def summarize_text(text):
    """Summarize transcript using Google Gemini."""
    model = genai.GenerativeModel("gemini-1.5-flash")

    MAX_CHARS = 5000
    chunks = [text[i:i+MAX_CHARS] for i in range(0, len(text), MAX_CHARS)]

    partial_summaries = []
    for chunk in chunks:
        resp = model.generate_content(f"Summarize this transcript part:\n\n{chunk}")
        partial_summaries.append(resp.text)

    final_summary = model.generate_content(
        "Combine these partial summaries into a concise overall summary:\n\n" + "\n".join(partial_summaries)
    )
    return final_summary.text

st.markdown('<div class="main-title">🎬 AI Video Summarizer</div>', unsafe_allow_html=True)
st.write("Upload a `.mp4` file and let AI give you a quick summary.")

uploaded_file = st.file_uploader("📂 Upload your video file", type=["mp4"])

if uploaded_file is not None:
    with st.spinner("🎵 Extracting audio from video..."):
        audio_path = extract_audio(uploaded_file)

    with st.spinner("📝 Transcribing video with Whisper..."):
        try:
            transcript = transcribe_audio_whisper(audio_path)
            st.markdown(f'<div class="box"><h3>📝 Transcript</h3><p>{transcript}</p></div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error during transcription: {e}")
            st.stop()

    with st.spinner("📌 Summarizing transcript with AI..."):
        try:
            summary = summarize_text(transcript)
            st.markdown(f'<div class="box"><h3>📌 Summary</h3><p>{summary}</p></div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error during summarization: {e}")

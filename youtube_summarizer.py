import streamlit as st
from pytube import YouTube
from haystack.nodes import PromptNode, PromptModel
from haystack.nodes.audio import WhisperTranscriber
from haystack.pipelines import Pipeline
from model_add import LlamaCPPInvocationLayer
import time
import os

# Streamlit page setup
st.set_page_config(page_title="YouTube Summarizer", layout="wide")

# ----------------- Utility Functions -----------------

def download_audio(url: str) -> str:
    """Download audio stream from YouTube video."""
    try:
        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        file_path = audio_stream.download(filename_prefix="audio_")
        return file_path
    except Exception as e:
        st.error(f"Error downloading YouTube audio: {e}")
        return None


def load_model(model_path: str):
    """Initialize Llama 2 model."""
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        return None
    model = PromptModel(
        model_name_or_path=model_path,
        invocation_layer_class=LlamaCPPInvocationLayer,
        use_gpu=False,
        max_length=512,
    )
    return model


def build_prompt_node(model):
    """Initialize Haystack prompt node for summarization."""
    template = "deepset/summarization"
    return PromptNode(model_name_or_path=model, default_prompt_template=template, use_gpu=False)


def summarize_audio(file_path: str, prompt_node):
    """Run Whisper transcription + Llama summarization pipeline."""
    whisper = WhisperTranscriber()
    pipe = Pipeline()
    pipe.add_node(component=whisper, name="whisper", inputs=["File"])
    pipe.add_node(component=prompt_node, name="prompt", inputs=["whisper"])
    return pipe.run(file_paths=[file_path])


# ----------------- Main Streamlit App -----------------

def main():
    st.title("🎥 YouTube Video Summarizer")
    st.markdown("#### Built with **Llama 2**, **Whisper**, **Haystack**, and **Streamlit**")

    with st.expander("ℹ️ About this App"):
        st.write(
            """
            This application automatically summarizes YouTube videos.
            Enter a video URL below and click **Submit**.
            The system downloads the audio, transcribes it using Whisper,
            and summarizes the transcript using Llama 2.
            """
        )

    youtube_url = st.text_input("Enter YouTube Video URL")

    model_path = st.text_input(
        "Enter Llama 2 model path",
        value="llama-2-7b-32k-instruct.Q4_K_S.gguf",
        help="Path to your local GGUF model file.",
    )

    if st.button("Submit") and youtube_url:
        start = time.time()

        st.info("Downloading and processing video...")
        file_path = download_audio(youtube_url)
        if not file_path:
            return

        model = load_model(model_path)
        if not model:
            return

        prompt_node = build_prompt_node(model)
        result = summarize_audio(file_path, prompt_node)

        end = time.time()
        elapsed = end - start

        # Layout: two columns
        col1, col2 = st.columns(2)
        with col1:
            st.video(youtube_url)
        with col2:
            st.header("Summary")
            try:
                summary_text = result["results"][0].split("\n\n[INST]")[0]
                st.success(summary_text)
            except Exception:
                st.warning("Summary extraction failed. Full output below:")
                st.write(result)
            st.caption(f"⏱️ Processing time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()

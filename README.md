YouTube Video Summarizer
A robust tool that automatically downloads audio from YouTube videos, transcribes it using Whisper, and generates a concise summary using the Llama 2 Large Language Model. It features a simple command-line interface and an optional Streamlit app for interactive use.

Features
Download YouTube audio and process it locally.

Speech-to-text transcription via OpenAI's Whisper model.

Automatic summarization using Llama 2 with Haystack.

Streamlit web interface for seamless user experience.

Requirements
Python 3.8+

pytube

haystack

OpenAI Whisper

Streamlit (for the web UI)

Llama 2 model weights (see model requirements below)

Install dependencies with:

bash
pip install -r requirements.txt
Model Requirements
Download the Llama 2 GGUF model (llama-2-7b-32k-instruct.Q4_K_S.gguf) and place it in your project directory. Adjust the path as needed in your settings.

Usage
Command Line Script
Run the main script to process a video:

bash
python summary.py
By default, it uses a sample YouTube video URL. Edit the file to process a different video.

Streamlit Web Application
Launch the interactive UI:

bash
streamlit run youtube_summarizer.py
Enter a YouTube video URL and the path to your local Llama 2 model.

Click submit to view the video and its summary.
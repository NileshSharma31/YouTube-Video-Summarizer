# Import necessary libraries for video/audio processing and summarization
from pytube import YouTube
from haystack.nodes import PromptNode, PromptModel
from haystack.nodes.audio import WhisperTranscriber
from haystack.pipelines import Pipeline
from model_add import LlamaCPPInvocationLayer

# Constants for model configuration
SUMMARY_PROMPT = "deepset/summarization"        # Summarization prompt template identifier
MODEL_PATH = "llama-2-7b-32k-instruct.Q4_K_S.gguf"  # Path to model weights
MODEL_MAX_LENGTH = 512                         # Maximum allowed summary length for the model

def youtube_to_audio(url: str, abr: str = '160kbps') -> str:
    """
    Downloads the audio stream of a YouTube video at the given URL and specified audio bitrate.
    Returns the local path to the downloaded audio file.

    Args:
        url (str): The YouTube video URL.
        abr (str): Audio bitrate as a filter, default is '160kbps'.

    Returns:
        str: Filepath of the downloaded audio.

    Raises:
        ValueError: If no stream matches the specified bitrate.
    """
    yt = YouTube(url)
    stream = yt.streams.filter(abr=abr).last()   # Get the last audio stream with specified bitrate
    if not stream:                               # Error handling for unavailable bitrate
        raise ValueError(f"No audio stream with {abr} available.")
    return stream.download()                      # Download and return the file path

def build_pipeline(model_path: str, max_length: int, use_gpu: bool = False) -> Pipeline:
    """
    Constructs a Haystack pipeline for audio transcription and summarization.

    Args:
        model_path (str): Path to the pre-trained language model weights.
        max_length (int): Maximum response length for model inference.
        use_gpu (bool): If True, use GPU for inference.

    Returns:
        Pipeline: Configured Haystack pipeline.
    """
    # Create an audio transcriber node (Whisper)
    whisper = WhisperTranscriber()
    # Prepare the summarization model node with LlamaCPP invocation
    model = PromptModel(
        model_name_or_path=model_path,
        invocation_layer_class=LlamaCPPInvocationLayer,
        use_gpu=use_gpu,
        max_length=max_length
    )
    # Wrap model in a prompt node for flexible prompt-based summarization
    prompt_node = PromptNode(
        model_name_or_path=model,
        default_prompt_template=SUMMARY_PROMPT,
        use_gpu=use_gpu
    )
    # Build the pipeline with two nodes: transcriber and summarizer
    pipeline = Pipeline()
    pipeline.add_node(component=whisper, name="whisper", inputs=["File"])
    pipeline.add_node(component=prompt_node, name="prompt", inputs=["whisper"])
    return pipeline

def main(url: str):
    """
    Orchestrates the fetching, transcription, and summarization of YouTube video audio.

    Args:
        url (str): The YouTube video URL to process.
    """
    try:
        # Step 1: Download audio from YouTube
        file_path = youtube_to_audio(url)
    except Exception as e:
        print(f"Failed to fetch audio: {e}")      # Print errors (e.g., no matching stream)
        return

    # Step 2: Build summarization pipeline using configured model
    pipeline = build_pipeline(MODEL_PATH, MODEL_MAX_LENGTH)
    # Step 3: Run pipeline to get transcription + summary results
    output = pipeline.run(file_paths=[file_path])
    results = output.get("results", [])
    if results:
        print(results)                            # Print full output results for debugging
        print(results[0].split("\n\n[INST]")[0])  # Extract and print the main summary
    else:
        print("No results produced by pipeline.") # Informative message for empty outputs

# Entry point for script execution — runs main on sample YouTube video
if __name__ == "__main__":
    YOUTUBE_URL = "https://www.youtube.com/watch?v=h5id4erwD4s"
    main(YOUTUBE_URL)

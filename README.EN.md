# Script for Video Transcription with Speaker Recognition

This Python script is used for the automatic transcription of video files (e.g., `.avi`, `.mp4`). Its main features are:

* **Audio Extraction**: Automatically extracts the audio track from the video file.

* **Speech-to-Text Transcription**: Uses the `WhisperX` model for precise speech-to-text conversion.

* **Speaker Recognition (Diarization)**: Identifies who is speaking at any given moment and assigns the corresponding lines to them.

* **Progress Saving**: Creates a working directory (`filename_work`) where it saves intermediate results. In case of an error or interruption, the script can be restarted, and it will resume from the last successful stage.

* **Save to `.docx`**: Generates a readable Word file with the final transcription, including speaker labels.

* **Configuration via `.env`**: Allows for easy management of settings using a `.env` file.

* **Advanced Optimizations**: Supports VAD (Voice Activity Detection) and the CTranslate2 engine for maximum performance on a CPU.

## Requirements

1. **Python**: Version 3.8 or newer.

2. **FFMPEG**: Essential for processing video files. It must be installed on the system and available in the `PATH` environment variable.

3. **NVIDIA Graphics Card (Optional, but recommended)**: Significantly speeds up the process.

## Installation

1. **Clone or download the repository** and navigate to the script's folder.

    ```bash
    git clone https://github.com/pk2/avi2text.git
    cd avi2text
    python -m venv .venv
    . .venv/bin/activate
    pip install -r requirements.txt
    ```


## Configuration

Before the first run, you need to configure the `.env` file.

1. **Create a `.env` file** in the same folder where the script is located.

2. **Copy and paste** the content below into the `.env` file and **fill in your token**:

    ```
    HUGGING_FACE_TOKEN="hf_TOKEN"
    DEFAULT_MODEL="large-v2"
    DEFAULT_SPEAKERS=2
    DEFAULT_LANGUAGE="pl"
    ```
3. **Fill in `HUGGING_FACE_TOKEN`**:

* Log in or register on [Hugging Face](https://huggingface.co/).

* Accept the terms of service on the model pages for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).

* Copy your token from **Settings -> Access Tokens** and paste it into the `.env` file.

## Usage

### Basic Usage

The script will use the settings from the `.env` file and automatically detect the best options for your hardware.
```bash
python3 avi2text.py "video.mp4"
```

### Advanced Options and Optimization

You can customize the script's behavior using flags:

`--liczba_mowcow NUMBER`: Specifies the exact number of speakers.

`--model MODEL`: Selects a different Whisper model (e.g., `medium`, `small`).

`--batch_size SIZE`: (GPU only) Sets the number of segments processed at once. Increase it (e.g., to 16, 32) if you have a GPU with a large amount of VRAM.

`--cpu_threads NUMBER`: (CPU only) Sets the number of processor threads. By default, it uses all available threads.

`--no-vad`: Disables the VAD filter (enabled by default).

`--compute_type TYPE`: Changes the computation type. To use CTranslate2 on a CPU and get a 4x speedup, use `--compute_type int8`.

Example of maximum optimization on a CPU:
```bash
python3 avi2text.py "video.avi" --model medium --compute_type int8
```


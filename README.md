# whisper-audio

Local YouTube → transcript → analysis pipeline. Everything runs on your machine: `yt-dlp` downloads the audio, `faster-whisper` (large-v3) transcribes it, and a local Ollama LLM produces a Markdown analysis. No cloud, no API keys.

## Prerequisites

### 1. FFmpeg (on PATH)
Whisper and `yt-dlp` need FFmpeg.

Windows:
```powershell
winget install Gyan.FFmpeg
```
Then open a new terminal so `PATH` refreshes. Verify:
```
ffmpeg -version
```

### 2. NVIDIA GPU + CUDA 12 / cuDNN 9 (optional, recommended)
`faster-whisper` 1.1+ uses CUDA 12 and cuDNN 9 for GPU inference. If you already have CUDA 12 and cuDNN 9 installed (via NVIDIA or a recent PyTorch install) you are done. Otherwise:
```
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12
```
If CUDA is not available, run the tool with `--device cpu --compute-type int8` (slower but works).

### 3. Ollama
Install from https://ollama.com/download, then in a separate terminal:
```
ollama serve
```
Pull the analysis model (first time only):
```
ollama pull qwen3.5:latest
```
Any other chat model you have pulled works too — pass it via `--ollama-model`.

## Install

Git Bash / WSL:
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

PowerShell / cmd:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration (`.env`)

Project-local settings live in `.env` (git-ignored). Copy `.env.example` to `.env` and fill in what you need — every variable is optional.

```bash
cp .env.example .env
```

Supported keys:

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | Hugging Face access token. Only needed for gated/private models. Get one at https://huggingface.co/settings/tokens. |
| `HF_ENDPOINT` | Alternate Hugging Face URL (e.g. `https://hf-mirror.com`). Leave empty for the official hub. |
| `HF_HOME` | Override the Hugging Face cache directory. |
| `OLLAMA_HOST` | Ollama server URL. Defaults to `http://localhost:11434`. |
| `OLLAMA_MODEL` | Default Ollama model for analysis. CLI flag `--ollama-model` overrides. |
| `WHISPER_MODEL` | Default Whisper model size. CLI flag `--model` overrides. |
| `WHISPER_DEVICE` | Default device (`cuda`, `cpu`, `auto`). CLI flag `--device` overrides. |
| `WHISPER_COMPUTE_TYPE` | CTranslate2 compute type. CLI flag `--compute-type` overrides. |
| `OUTPUT_DIR` | Default output root directory. CLI flag `--output-dir` overrides. |

Values in `.env` are loaded by `config.py` at startup and never override variables that are already set in your real environment. CLI flags always override `.env`.

The Whisper `large-v3` weights (~3 GB) download automatically on first run and are cached under `~/.cache/huggingface/hub`.

## Usage

```
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

The folder is named from the video title (spaces replaced with `_`, Windows-forbidden characters stripped). Output lands in `output/<Video_Title>/`:
- `audio.m4a` — downloaded source audio
- `transcript.txt` — plain-text transcript
- `transcript.srt` — timestamped subtitles
- `analysis.md` — Summary, Key Points, Topics, Notable Quotes

### Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--audio <path>` | – | Re-transcribe an existing local audio file; skips download. Writes transcript + analysis next to the audio. |
| `--model` | `large-v3` | Whisper model size |
| `--device` | `cuda` | `cuda`, `cpu`, or `auto` |
| `--compute-type` | `float16` | e.g. `float16`, `int8_float16`, `int8` |
| `--ollama-model` | `qwen3.5:latest` | Any model you have pulled in Ollama |
| `--skip-analysis` | off | Stop after writing the transcript |
| `--output-dir` | `output` | Root folder for per-video output |

### Examples

CPU-only, smaller model:
```
python main.py "https://youtu.be/x7X9w_GIm1s" --device cpu --compute-type int8 --model medium
```

Transcript only (no Ollama needed):
```
python main.py "https://youtu.be/x7X9w_GIm1s" --skip-analysis
```

Different analysis model:
```
python main.py "https://youtu.be/x7X9w_GIm1s" --ollama-model llama3.1:8b
```

Re-transcribe an already-downloaded audio file (skips YouTube entirely):
```
python main.py --audio output/VE7IaVrsH0s/audio.m4a
```
Transcript and analysis are written next to the audio, overwriting any stale ones.

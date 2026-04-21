import config  # noqa: F401  # loads .env before HF/Ollama imports

import argparse
import os
import sys
from pathlib import Path

import analyzer
import transcriber
from downloader import download_audio


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a YouTube video, transcribe it locally with Whisper, and analyze it."
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--model",
        default=os.environ.get("WHISPER_MODEL") or "large-v3",
        help="Whisper model size (default from WHISPER_MODEL or large-v3)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("WHISPER_DEVICE") or "cuda",
        choices=["cuda", "cpu", "auto"],
    )
    parser.add_argument(
        "--compute-type",
        default=os.environ.get("WHISPER_COMPUTE_TYPE") or "float16",
        help="CTranslate2 compute_type, e.g. float16, int8_float16, int8",
    )
    parser.add_argument(
        "--ollama-model",
        default=os.environ.get("OLLAMA_MODEL") or "qwen3.5:latest",
    )
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR") or "output",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)

    print(f"Downloading audio from {args.url} ...")
    try:
        audio_path, slug, title = download_audio(args.url, output_root)
    except Exception as e:
        print(f"ERROR: download failed: {e}", file=sys.stderr)
        return 1
    video_dir = audio_path.parent
    print(f"Downloaded: {title}")
    print(f"  audio: {audio_path}")

    device = args.device
    compute_type = args.compute_type
    if device == "auto":
        device, compute_type = _auto_device(compute_type)

    print(f"Loading Whisper model '{args.model}' on {device} ({compute_type}) ...")
    try:
        t = transcriber.Transcriber(
            model_size=args.model, device=device, compute_type=compute_type
        )
    except Exception as e:
        print(f"ERROR: failed to load Whisper model: {e}", file=sys.stderr)
        print(
            "Hint: if CUDA/cuDNN is missing, re-run with --device cpu --compute-type int8",
            file=sys.stderr,
        )
        return 1

    print("Transcribing ...")
    text, segments = t.transcribe(audio_path)
    txt_path = video_dir / "transcript.txt"
    srt_path = video_dir / "transcript.srt"
    transcriber.write_txt(text, txt_path)
    transcriber.write_srt(segments, srt_path)
    print(f"Transcribed {len(segments)} segments")
    print(f"  transcript: {txt_path}")
    print(f"  subtitles:  {srt_path}")

    if args.skip_analysis:
        print("Skipping analysis (--skip-analysis).")
        return 0

    print(f"Analyzing with Ollama model '{args.ollama_model}' ...")
    try:
        analysis_md = analyzer.analyze(text, model=args.ollama_model)
    except Exception as e:
        print(f"ERROR: analysis failed: {e}", file=sys.stderr)
        print(
            "Hint: ensure Ollama is running ('ollama serve') and the model is pulled "
            f"('ollama pull {args.ollama_model}'), or re-run with --skip-analysis.",
            file=sys.stderr,
        )
        return 1

    analysis_path = video_dir / "analysis.md"
    analysis_path.write_text(analysis_md, encoding="utf-8")
    print(f"  analysis:   {analysis_path}")
    return 0


def _auto_device(requested_compute_type: str) -> tuple[str, str]:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", requested_compute_type
    except Exception:
        pass
    return "cpu", "int8"


if __name__ == "__main__":
    sys.exit(main())

import os
import site
import sys
from dataclasses import dataclass
from pathlib import Path


def _register_nvidia_dll_dirs() -> None:
    if sys.platform != "win32":
        return
    subdirs = ("cudnn/bin", "cublas/bin", "cuda_nvrtc/bin", "cuda_runtime/bin")
    paths_added: list[str] = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        nvidia_root = Path(sp) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for sub in subdirs:
            p = nvidia_root / sub
            if p.is_dir():
                p_str = str(p)
                os.add_dll_directory(p_str)
                paths_added.append(p_str)
    # Also prepend to PATH so the Windows loader finds transitively-loaded DLLs
    # (add_dll_directory alone isn't enough for DLL-loads-DLL chains on Windows).
    if paths_added:
        os.environ["PATH"] = os.pathsep.join(
            paths_added + [os.environ.get("PATH", "")]
        )


_register_nvidia_dll_dirs()

from faster_whisper import WhisperModel  # noqa: E402


@dataclass
class Segment:
    start: float
    end: float
    text: str


class Transcriber:
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
    ) -> None:
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: Path) -> tuple[str, list[Segment]]:
        segments_iter, _info = self.model.transcribe(
            str(audio_path),
            beam_size=5,
            vad_filter=True,
        )
        segments = [
            Segment(start=s.start, end=s.end, text=s.text.strip())
            for s in segments_iter
        ]
        text = " ".join(s.text for s in segments).strip()
        return text, segments


def write_txt(text: str, path: Path) -> None:
    path.write_text(text, encoding="utf-8")


def write_srt(segments: list[Segment], path: Path) -> None:
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_fmt_ts(seg.start)} --> {_fmt_ts(seg.end)}")
        lines.append(seg.text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms_total = int(round(seconds * 1000))
    hours, rem = divmod(ms_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

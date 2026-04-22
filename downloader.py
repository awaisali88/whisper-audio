import re
from pathlib import Path

from yt_dlp import YoutubeDL

# Characters Windows forbids in file/folder names, plus control chars.
_WIN_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _slugify(title: str) -> str:
    """Make `title` usable as a folder name on Windows and Unix while keeping
    it readable: preserves Unicode and ordinary punctuation, strips only
    Windows-invalid characters, collapses whitespace to single underscores,
    and removes the `%` yt-dlp uses for output templates."""
    s = _WIN_INVALID.sub("", title)
    s = s.replace("%", "")
    s = re.sub(r"\s+", "_", s).strip(". _")
    return s or "video"


def download_audio(url: str, out_dir: Path) -> tuple[Path, str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Probe metadata first so we slugify the title ourselves rather than
    # relying on yt-dlp's `restrictfilenames`, which strips all non-ASCII.
    with YoutubeDL({"quiet": True, "no_warnings": True, "noprogress": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    title = info.get("title") or info["id"]
    slug = _slugify(title)

    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio",
        "outtmpl": str(out_dir / slug / "audio.%(ext)s"),
        "windowsfilenames": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "overwrites": False,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_path = Path(ydl.prepare_filename(info))

    return audio_path, slug, title

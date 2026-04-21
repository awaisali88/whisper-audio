from pathlib import Path

from yt_dlp import YoutubeDL


def download_audio(url: str, out_dir: Path) -> tuple[Path, str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio",
        "outtmpl": str(out_dir / "%(title)s" / "audio.%(ext)s"),
        "restrictfilenames": True,  # spaces -> "_", strips Windows-forbidden chars
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "overwrites": False,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_path = Path(ydl.prepare_filename(info))

    slug = audio_path.parent.name
    return audio_path, slug, info.get("title", info["id"])

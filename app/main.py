import os, tempfile, subprocess, shutil, re
from pathlib import Path
from typing import Optional
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Video to SRT")
BASE = Path(__file__).parent

app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home():
    return (BASE / "static" / "index.html").read_text(encoding="utf-8")

def srt_time(seconds: float) -> str:
    ms = max(0, int(round(float(seconds) * 1000)))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def make_srt(segments):
    out = []
    for i, seg in enumerate(segments, 1):
        start = float(seg["start"])
        end = max(start + 0.15, float(seg["end"]))
        text = re.sub(r"\s+", " ", str(seg["text"])).strip()
        if text:
            out.append(f"{i}\n{srt_time(start)} --> {srt_time(end)}\n{text}")
    return "\n\n".join(out) + ("\n" if out else "")

def extract_audio_chunks(video_path: str, output_dir: str, chunk_seconds: int = 480):
    """Extract the video's original audio into 8-minute MP3 chunks.

    Chunking avoids provider per-request upload limits. There is no fixed total
    extracted-audio size limit in this app; practical limits are server disk,
    upload size, runtime, and provider quotas.
    """
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("FFmpeg and FFprobe are required on the server.")
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True
    )
    try:
        duration = float(probe.stdout.strip())
    except (ValueError, TypeError):
        raise RuntimeError("Could not determine video duration.")
    if duration <= 0:
        raise RuntimeError("The video has no readable audio duration.")
    chunks = []
    start = 0.0
    index = 0
    while start < duration:
        path = os.path.join(output_dir, f"audio_{index:04d}.mp3")
        cmd = [ffmpeg, "-y", "-ss", str(start), "-i", video_path, "-t",
               str(chunk_seconds), "-vn", "-ac", "1", "-ar", "16000",
               "-b:a", "64k", "-f", "mp3", path]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0 or not os.path.exists(path) or os.path.getsize(path) == 0:
            raise RuntimeError("Could not extract an audio chunk. Please check the video file.")
        chunks.append((path, start))
        start += chunk_seconds
        index += 1
    return chunks

async def groq_transcribe(audio_path: str, key: str, language: str):
    data = {"model": "whisper-large-v3", "response_format": "verbose_json",
            "timestamp_granularities[]": "segment", "temperature": "0"}
    if language != "auto":
        data["language"] = language
    async with httpx.AsyncClient(timeout=600) as client:
        with open(audio_path, "rb") as f:
            r = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {key}"},
                data=data,
                files={"file": ("audio.mp3", f, "audio/mpeg")}
            )
    if r.status_code >= 400:
        raise RuntimeError(f"Groq API error ({r.status_code}): {r.text[:500]}")
    payload = r.json()
    return [{"start": x["start"], "end": x["end"], "text": x["text"]}
            for x in payload.get("segments", [])]

def offset_seconds(value):
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    if s.endswith("s"):
        s = s[:-1]
    try:
        return float(s)
    except ValueError:
        return 0.0

async def gemini_transcribe(audio_path: str, key: str, language: str):
    try:
        from google import genai
    except ImportError:
        raise RuntimeError("Gemini SDK missing on server. Install requirements.txt.")
    client = genai.Client(api_key=key)
    uploaded = client.files.upload(file=audio_path)
    config = {
        "transcription_config": {
            "mode": {"type": "verbatim", "timestamp_granularities": ["word"]}
        }
    }
    interaction = client.interactions.create(
        model="gemini-3.5-transcribe",
        input=[{"type": "audio", "uri": uploaded.uri, "mime_type": uploaded.mime_type}],
        generation_config=config
    )
    words = []
    for step in (getattr(interaction, "steps", None) or []):
        for content in (getattr(step, "content", None) or []):
            for ann in (getattr(content, "annotations", None) or []):
                if getattr(ann, "type", None) == "word_info":
                    words.append({
                        "start": offset_seconds(getattr(ann, "start_offset", 0)),
                        "end": offset_seconds(getattr(ann, "end_offset", 0)),
                        "text": getattr(ann, "text", "") or getattr(ann, "word", "")
                    })
    if not words:
        # Fail clearly rather than fabricate timestamp timings.
        raise RuntimeError("Gemini returned transcript text but no word timestamps. Try Groq for SRT export.")
    # Turn word-level annotations into readable subtitle phrases.
    segments, group = [], []
    for w in words:
        if not w["text"]:
            continue
        group.append(w)
        joined = " ".join(x["text"] for x in group).strip()
        if len(group) >= 7 or len(joined) >= 42 or re.search(r"[.!?။]$", joined):
            segments.append({"start": group[0]["start"], "end": group[-1]["end"], "text": joined})
            group = []
    if group:
        segments.append({"start": group[0]["start"], "end": group[-1]["end"],
                         "text": " ".join(x["text"] for x in group).strip()})
    return segments

@app.post("/api/transcribe")
async def transcribe(
    video: UploadFile = File(...),
    engine: str = Form(...),
    api_key: str = Form(...),
    language: str = Form("my")
):
    if engine not in ("groq", "gemini"):
        raise HTTPException(400, "Unsupported transcription engine.")
    if not api_key.strip():
        raise HTTPException(400, "Please enter your API key.")
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "input" + suffix)
        audio_path = os.path.join(tmp, "audio.mp3")
        size = 0
        with open(video_path, "wb") as dst:
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > 500 * 1024 * 1024:
                    raise HTTPException(413, "Video is too large (500 MB max).")
                dst.write(chunk)
        try:
            chunks = extract_audio_chunks(video_path, tmp, chunk_seconds=480)
            segments = []
            for audio_path, offset in chunks:
                if engine == "groq":
                    chunk_segments = await groq_transcribe(audio_path, api_key.strip(), language)
                else:
                    chunk_segments = await gemini_transcribe(audio_path, api_key.strip(), language)
                for seg in chunk_segments:
                    segments.append({
                        "start": float(seg["start"]) + offset,
                        "end": float(seg["end"]) + offset,
                        "text": seg["text"]
                    })
            if not segments:
                raise RuntimeError("No speech segments found in this video.")
            return {"segments": segments, "srt": make_srt(segments), "count": len(segments)}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, str(e))

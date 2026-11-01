# Video to SRT

A small online web app that extracts the original audio from an uploaded video, transcribes it with Groq Whisper or Gemini 3.5 Transcribe, and creates an editable `.srt` subtitle file.

## Features
- Upload MP4/MOV/MKV/WEBM videos (server upload limit: 500 MB)
- Burmese, English, Thai, or automatic language selection
- Groq Whisper with segment timestamps
- Gemini 3.5 Transcribe with word timestamps grouped into subtitle lines
- Edit the generated SRT in the browser and download it
- API keys are submitted per request and are not written to disk by this app

## Run locally
Requires Python 3.11+ and FFmpeg.

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open http://localhost:8000

## Deploy online with Render
1. Push this project to a GitHub repository.
2. In Render, create a new Blueprint and select the repository (it reads `render.yaml`).
3. Deploy the Docker service.
4. Open the generated public URL.

## API keys
- Groq: https://console.groq.com/keys
- Gemini: https://aistudio.google.com/app/apikey

Enter the key in the app when running a transcription. The app does not persist the key. Use HTTPS in production.

## Important MVP limits
- Audio is split into 8-minute chunks automatically, so a single provider upload-size limit no longer blocks longer videos. There is no fixed total extracted-audio size limit in the app; practical limits still include the 500 MB video upload cap, server storage/runtime, and provider quotas.
- The Gemini integration requires the current `google-genai` SDK and the Gemini Interactions API with `gemini-3.5-transcribe`.
- Always review Burmese spellings, names, and subtitle timing before importing into CapCut.
- This starter app has no user accounts, job queue, rate limiting, or persistent history. Add these before opening a public service to many users.

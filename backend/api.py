from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import pipeline

app = FastAPI(
    title="Neural Speech-to-Text API",
    description="Optimized backend for speech recognition using Whisper models.",
    version="1.0.0"
)

# Required: Allow Angular Frontend (localhost:4200) to communicate without CORS issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for local dev. In production set to ["http://localhost:4200"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional GPU validation
device = 0 if torch.cuda.is_available() else -1
print("Initializing STT Pipeline...")
print(f"Target Device: {'CUDA (GPU)' if device == 0 else 'CPU'}")

# Pipeline Initialization
# For local dev, we load the base model. To load your fine-tuned model, you'd merge the LoRA weights first!
stt_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    chunk_length_s=30,
    device=device,
)
print("Pipeline loaded successfully.")

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "Backend API is fully operational"}

@app.post("/api/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Receives an audio blob sent via multipart/form-data from the frontend."""
    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Unsupported file format sent. Expected an audio file.")
    
    try:
        # Read the file byte stream natively
        audio_bytes = await audio_file.read()
        
        # HuggingFace pipeline processes raw audio bytes automatically (via ffmpeg bindings under the hood)
        result = stt_pipeline(audio_bytes)
        
        return {
            "status": "success",
            "transcript": result["text"].strip()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")

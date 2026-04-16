from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import os

app = FastAPI(
    title="Neural STT API (With Custom LoRA)",
    description="Backend for speech recognition merged with your privately trained LoRA weights.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = 0 if torch.cuda.is_available() else -1
print("Initializing Custom STT Pipeline...")

# Trỏ đến thư mục lora bạn copy từ Server về
LORA_PATH = "../ml/best_whisper_lora"
# Phải TRÙNG KHỚP với base_model bạn đã dùng ở Server
BASE_MODEL_ID = "openai/whisper-large-v3-turbo" 

if os.path.exists(LORA_PATH):
    print("Found LoRA weights. Merging with Base Model...")
    # 1. Tải Base Model
    base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID)
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    
    # 2. Hợp thể (Merge)
    model_voi_lora = PeftModel.from_pretrained(base_model, LORA_PATH)
    merged_model = model_voi_lora.merge_and_unload()
    
    # 3. Gắn Model hoàn chỉnh
    stt_pipeline = pipeline(
        "automatic-speech-recognition",
        model=merged_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        device=device,
    )
    print("Merged successfully!")
else:
    print(f"[{LORA_PATH}] NOT FOUND. Falling back to default Whisper...")
    stt_pipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        chunk_length_s=30,
        device=device,
    )

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "Custom LoRA API is operational"}

@app.post("/api/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Expected an audio file.")
    try:
        audio_bytes = await audio_file.read()
        result = stt_pipeline(audio_bytes)
        return {
            "status": "success",
            "transcript": result["text"].strip()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

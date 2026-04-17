import os
import unicodedata

import torch
import evaluate
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

"""
OPTIMIZED SCRIPT FOR RTX 5060 Ti (~16GB VRAM)
=====================================================
Tối ưu cho card 16GB VRAM:

1. 8-bit quantization (bitsandbytes) → giảm ~50% VRAM cho model weights
2. Batch size nhỏ (2) + gradient accumulation (16) → effective batch = 32
3. LoRA rank 32 + target 4 modules → cân bằng chất lượng và VRAM
4. WER metric thay vì loss → đánh giá chính xác chất lượng STT
5. Warmup + Cosine scheduler → ổn định training
6. Vietnamese text normalization
7. Dùng test set có sẵn thay vì tự split
8. 8-bit optimizer (adamw_bnb_8bit) → tiết kiệm thêm VRAM
9. Gradient checkpointing → giảm ~60% VRAM cho activations
"""

# ========================
# CONFIGURATION
# ========================
MODEL_ID = "openai/whisper-large-v3-turbo"
TRAIN_DATA_DIR = "o:/brother_web/training/train"
TEST_DATA_DIR = "o:/brother_web/training/test"
OUTPUT_DIR = "./whisper-lora-large-finetuned"
FINAL_MODEL_DIR = "./best_whisper_lora"
SEED = 42

# Cố định seed để kết quả reproducible
set_seed(SEED)

# ========================
# 1. HELPER FUNCTIONS
# ========================

def normalize_vietnamese(text):
    """Chuẩn hóa Unicode NFC cho tiếng Việt và loại bỏ khoảng trắng thừa."""
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.strip().split())
    return text


def load_vivos_dataset(data_dir):
    """Load dataset từ thư mục VIVOS format (prompts.txt + waves/)."""
    prompts_path = os.path.join(data_dir, "prompts.txt")
    waves_dir = os.path.join(data_dir, "waves")
    
    data_list = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                file_id, sentence = parts
                speaker_id = file_id.split("_")[0]
                audio_path = os.path.join(waves_dir, speaker_id, f"{file_id}.wav")
                if os.path.exists(audio_path):
                    data_list.append({"audio": audio_path, "sentence": sentence})
    
    print(f"  -> Loaded {len(data_list)} samples from {data_dir}")
    return Dataset.from_list(data_list)

# ========================
# 2. LOAD DATASETS (Dùng test set có sẵn)
# ========================
print("Loading local training dataset...")
train_dataset = load_vivos_dataset(TRAIN_DATA_DIR)
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))

print("Loading local test/evaluation dataset...")
eval_dataset = load_vivos_dataset(TEST_DATA_DIR)
eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=16000))

# ========================
# 3. PROCESSOR SETUP
# ========================
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="Vietnamese", task="transcribe")


def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    # Normalize text tiếng Việt trước khi tokenize
    sentence = normalize_vietnamese(batch["sentence"])
    batch["labels"] = processor.tokenizer(sentence).input_ids
    return batch


print("Preprocessing train dataset...")
train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names, num_proc=4)

print("Preprocessing eval dataset...")
eval_dataset = eval_dataset.map(prepare_dataset, remove_columns=eval_dataset.column_names, num_proc=4)

# ========================
# 4. MODEL LOADING — 8-bit quantization (tiết kiệm VRAM cho 16GB)
# ========================
print(f"Loading {MODEL_ID} in 8-bit quantization...")
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID,
    load_in_8bit=True,            # 8-bit quantization → giảm ~50% VRAM model weights
    device_map="auto"
)
model.config.use_cache = False    # Tắt cache để dùng Gradient Checkpointing
model = prepare_model_for_kbit_training(model)  # Chuẩn bị model cho LoRA + quantization

# ========================
# 5. LoRA CONFIGURATION (Rank 32, 4 modules — cân bằng cho 16GB VRAM)
# ========================
print("Applying LoRA parameters...")
lora_config = LoraConfig(
    r=32,                    # Rank 32: cân bằng chất lượng và VRAM
    lora_alpha=64,           # Alpha = 2x rank
    target_modules=[         # Target attention modules chính
        "q_proj", "k_proj", "v_proj", "out_proj"
    ],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ========================
# 6. WER METRIC
# ========================
wer_metric = evaluate.load("wer")


def compute_metrics(pred):
    """Tính Word Error Rate — chuẩn vàng cho Speech-to-Text."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str = [normalize_vietnamese(s) for s in pred_str]
    label_str = [normalize_vietnamese(s) for s in label_str]

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# ========================
# 7. TRAINING ARGUMENTS — Tối ưu cho RTX 5060 Ti 16GB VRAM
# ========================
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,       # Batch nhỏ cho 16GB VRAM
    per_device_eval_batch_size=1,        # Eval nhỏ vì predict_with_generate tốn thêm VRAM
    gradient_accumulation_steps=16,      # Effective batch size = 2 x 16 = 32 (giữ nguyên)
    gradient_checkpointing=True,         # Giảm ~60% VRAM cho activations
    learning_rate=1e-4,                  # Phù hợp cho fine-tune large model
    warmup_steps=500,                    # Tăng LR từ từ trong 500 bước đầu
    lr_scheduler_type="cosine",          # Giảm LR mượt mà
    num_train_epochs=5,                  # 5 epochs + early stopping
    fp16=True,                           # fp16 mixed precision (8-bit model không hỗ trợ bf16)
    predict_with_generate=True,          # Bắt buộc để tính WER
    generation_max_length=225,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    logging_steps=25,                    # Log thường xuyên hơn để theo dõi
    remove_unused_columns=False,
    label_names=["labels"],
    optim="adamw_bnb_8bit",              # 8-bit optimizer → tiết kiệm ~30% VRAM cho optimizer states
    load_best_model_at_end=True,
    metric_for_best_model="wer",         # WER thay vì loss
    greater_is_better=False,
    save_total_limit=3,                  # Giới hạn 3 checkpoint
    dataloader_num_workers=4,            # Multi-threaded data loading
    dataloader_pin_memory=True,          # Pin memory cho transfer CPU→GPU nhanh hơn
    seed=SEED,
)

# ========================
# 8. DATA COLLATOR
# ========================
class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# ========================
# 9. TRAINER
# ========================
print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# ========================
# 10. TRAIN -> SAVE
# ========================
# Xóa cache GPU trước khi bắt đầu training
torch.cuda.empty_cache()

print("=" * 60)
print("  RTX 5060 Ti 16GB — OPTIMIZED TRAINING")
print("=" * 60)
print(f"  Model:          {MODEL_ID}")
print(f"  VRAM:           ~16GB (8-bit quantization + gradient checkpoint)")
print(f"  Batch size:     {training_args.per_device_train_batch_size} x {training_args.gradient_accumulation_steps} = {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Learning rate:  {training_args.learning_rate}")
print(f"  Scheduler:      cosine (warmup {training_args.warmup_steps} steps)")
print(f"  Epochs:         {training_args.num_train_epochs} (early stopping patience=3)")
print(f"  LoRA rank:      {lora_config.r}, targets: {lora_config.target_modules}")
print(f"  Optimizer:      adamw_bnb_8bit")
print(f"  Metric:         WER (Word Error Rate)")
print("=" * 60)

trainer.train()

print("Saving final LoRA adapter + processor...")
model.save_pretrained(FINAL_MODEL_DIR)
processor.save_pretrained(FINAL_MODEL_DIR)
print(f"Training completed! Model saved to '{FINAL_MODEL_DIR}'")
print("Copy the folder back to your local machine for deployment.")

import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

"""
HIGH-PERFORMANCE SCRIPT FOR SERVER (e.g. RTX 5060 Ti / 16GB+ VRAM)
Upgraded to whisper-large-v3-turbo with higher batch sizes.
"""

MODEL_ID = "openai/whisper-large-v3-turbo"

import os
from datasets import Dataset, Audio

# 1. Load Local Custom Dataset
print("Loading local training dataset for Server...")
def load_vivos_dataset(data_dir="o:/brother_web/training/train"):
    prompts_path = os.path.join(data_dir, "prompts.txt")
    waves_dir = os.path.join(data_dir, "waves")
    
    data_list = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                file_id, sentence = parts
                speaker_id = file_id.split("_")[0]
                audio_path = os.path.join(waves_dir, speaker_id, f"{file_id}.wav")
                if os.path.exists(audio_path):
                    data_list.append({"audio": audio_path, "sentence": sentence})
                
    return Dataset.from_list(data_list)

dataset = load_vivos_dataset()
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# 2. Processor Setup
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="Vietnamese", task="transcribe")

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names, num_proc=4) # Tăng num_proc nếu CPU mạnh

# 3. Model Loading (Vẫn dùng 8-bit để đảm bảo an toàn VRAM, nhưng bạn có thể thử tắt load_in_8bit nếu VRAM dư dả)
print(f"Loading {MODEL_ID}...")
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    load_in_8bit=True,   
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False  # Bắt buộc tắt cache để dùng được Gradient Checkpointing

# 4. LoRA Configuration
print("Applying LoRA parameters...")
config = LoraConfig(
    r=32, # Tăng rank lên 32 để mô hình có không gian học tốt nhất
    lora_alpha=64, # Alpha luôn giữ bằng 2 lần rank
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

# 5. Training Setup
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-lora-large-finetuned",
    per_device_train_batch_size=2,       # Giảm gánh nặng cho VRAM
    gradient_accumulation_steps=8,       # Bù trừ lại bằng cách cộng gộp gradient (2 x 8 = 16) giúp model vẫn học cực thông minh
    gradient_checkpointing=True,         # BẬT TÍNH NĂNG NÀY: Giảm 50%-70% VRAM tiêu thụ, giúp chạy mượt mà không bị Crash OOM
    learning_rate=1e-3,
    num_train_epochs=3,
    fp16=True,                           
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps", # Phải trùng với evaluation_strategy cho Early Stopping
    save_steps=200,
    logging_steps=50,
    remove_unused_columns=False,
    label_names=["labels"],
    optim="adamw_bnb_8bit",              
    load_best_model_at_end=True, # Load lại trọng số tốt nhất trước khi kết thúc
    metric_for_best_model="loss", # Đo lường sự thay đổi thông qua độ sai lệch (Loss)
    greater_is_better=False, # Trọng số tốt hơn nghĩa là Loss phải thấp hơn
)

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

# Để có mốc đo lường sự thay đổi (evaluation) thì ta cần trích 10% dữ liệu ra làm file đối chiếu (Test)
print("Splitting dataset into train and validation sets...")
dataset_split = dataset.train_test_split(test_size=0.1)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_split["train"],
    eval_dataset=dataset_split["test"],
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # Nếu 3 chu kỳ (3 * 200 bước) mà trọng số không tốt hơn thì sẽ DỪNG NGAY!
)

# 6. Train -> Save
print("Starting high-performance training...")
trainer.train()

print("Saving final LoRA adapter...")
model.save_pretrained("./best_whisper_lora")
print("Training successfully completed! Copy the 'best_whisper_lora' folder back to your local machine.")

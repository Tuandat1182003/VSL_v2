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
Optimization strategies applied for ~4GB VRAM GPU:
1. 8-bit Quantization (bitsandbytes via load_in_8bit=True)
2. LoRA (Low-Rank Adaptation) targetting attention weights
3. Gradient Accumulation (Gradient accumulation = 4, per device batch sz = 4 -> virtual batch sz = 16)
4. Mixed Precision (fp16=True)
5. 8-bit Optimizer (adamw_bnb_8bit)
"""

import os
from datasets import Dataset, Audio

# 1. Load Local Custom Dataset
print("Loading local training dataset...")
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
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Vietnamese", task="transcribe")

def prepare_dataset(batch):
    audio = batch["audio"]
    # Compute log-Mel input features and encode target text
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names, num_proc=2)

# 3. Memory-Optimized Model Loading
print("Loading model in 8-bit precision...")
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small", 
    load_in_8bit=True,   # Requires bitsandbytes library
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# 4. LoRA Configuration
print("Applying LoRA parameters...")
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

# 5. Training Setup
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-lora-finetuned",
    per_device_train_batch_size=4,       
    gradient_accumulation_steps=4,       
    learning_rate=1e-3,
    num_train_epochs=3,
    fp16=True,                           
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps", # Rất quan trọng khi chạy EarlyStopping
    save_steps=100,
    logging_steps=25,
    remove_unused_columns=False,
    label_names=["labels"],
    optim="adamw_bnb_8bit",
    load_best_model_at_end=True, # Tự giữ lại phiên bản tốt nhất
    metric_for_best_model="loss", # Căn cứ vào độ sai lệch
    greater_is_better=False,
)

class DataCollatorSpeechSeq2SeqWithPadding:
    """A custom collator for padding features and labels."""
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

print("Splitting dataset into train and validation sets...")
dataset_split = dataset.train_test_split(test_size=0.1)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_split["train"],
    eval_dataset=dataset_split["test"],
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)] # Dừng sớm nếu qua 4 lần check mà không cải thiện độ Loss
)

# 6. Train -> Save
print("Starting training...")
trainer.train()

print("Saving final LoRA adapter...")
model.save_pretrained("./best_whisper_lora")
print("Training successfully completed!")

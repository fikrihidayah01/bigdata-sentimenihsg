import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import evaluate
import os

print("Memulai proses persiapan Fine-Tuning IndoBERT...")

# 1. Load Data
try:
    df = pd.read_csv("output/sentiment_result.csv")
    print(f"Berhasil meload data dengan {len(df)} baris.")
except Exception as e:
    print(f"Error membaca data: {e}")
    exit()

# Gunakan sentiment_lexicon sebagai ground truth (weak supervision)
df = df.dropna(subset=['text', 'sentiment_lexicon'])

# Label mapping
label_map = {"Positif": 0, "Netral": 1, "Negatif": 2}
df['label'] = df['sentiment_lexicon'].map(label_map)

# Buat Dataset HuggingFace
dataset = Dataset.from_pandas(df[['text', 'label']])
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# 2. Load Tokenizer & Model
model_name = "mdhugol/indonesia-bert-sentiment-classification"
print(f"Meload model {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, ignore_mismatched_sizes=True)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

print("Melakukan Tokenisasi Dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. Metrik Evaluasi
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir="./models/indobert_finetuned",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2, # Menggunakan 2 epoch agar tidak terlalu lama
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

print("\n===========================================")
print("MEMULAI FINE-TUNING INDOBERT")
print("Catatan: Tanpa GPU (CUDA), proses ini bisa memakan waktu 10-30 menit.")
print("===========================================\n")

trainer.train()

print("\nMenyimpan model hasil Fine-Tuning ke folder models/indobert_finetuned ...")
trainer.save_model("./models/indobert_finetuned")
tokenizer.save_pretrained("./models/indobert_finetuned")

print("Proses Fine-Tuning Selesai Sukses!")

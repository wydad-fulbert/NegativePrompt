from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
import torch

MODEL_NAME = "google/flan-t5-large"
OUTPUT_DIR = "/kaggle/working/t5_lora_np_robust_large_fixed"

MAX_INPUT = 192
MAX_TARGET = 32

dataset = load_dataset("json", data_files={
    "train": "lora_train.jsonl",
})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q", "v"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def preprocess(examples):
    model_inputs = tokenizer(
        examples["source"],
        max_length=MAX_INPUT,
        truncation=True,
        padding=False
    )

    labels = tokenizer(
        text_target=examples["target"],
        max_length=MAX_TARGET,
        truncation=True,
        padding=False
    )

    cleaned_labels = []
    for seq in labels["input_ids"]:
        cleaned_labels.append([
            token if token != tokenizer.pad_token_id else -100
            for token in seq
        ])

    model_inputs["labels"] = cleaned_labels
    return model_inputs

tokenized = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="longest",
    label_pad_token_id=-100
)

args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    num_train_epochs=1,
    logging_steps=50,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    report_to="none",
    dataloader_num_workers=2
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    data_collator=data_collator,
    processing_class=tokenizer,
)

trainer.train()
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("LoRA training finished.")

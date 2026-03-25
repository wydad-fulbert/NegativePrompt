from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import torch

MODEL_NAME = "google/flan-t5-large"   # si OOM, remplace par "google/flan-t5-base"
OUTPUT_DIR = "t5_lora_np_robust"

MAX_INPUT = 256
MAX_TARGET = 64

dataset = load_dataset("json", data_files={
    "train": "lora_train.jsonl",
})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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
        padding="max_length"
    )
    labels = tokenizer(
        text_target=examples["target"],
        max_length=MAX_TARGET,
        truncation=True,
        padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=2,
    logging_steps=20,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    report_to="none"
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
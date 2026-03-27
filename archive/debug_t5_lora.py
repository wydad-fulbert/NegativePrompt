import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

BASE_MODEL = "google/flan-t5-large"
LORA_DIR = "/kaggle/working/t5_lora_np_robust_large_fixed"

base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

lora_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
lora_base = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
lora_model = PeftModel.from_pretrained(lora_base, LORA_DIR)

if torch.cuda.is_available():
    base_model = base_model.to("cuda")
    lora_model = lora_model.to("cuda")

def generate(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=192)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

tests = [
    "Instruction: Determine whether a movie review is positive or negative.\n\nInput: This film was excellent and moving.\nAnswer:",
    "Instruction: Translate the word into French.\n\nInput: hello\nAnswer:",
    "Instruction: Count the number of objects described in the input.\n\nInput: I have 3 apples and 2 bananas.\nAnswer:"
]

for t in tests:
    print("\nPROMPT:", t)
    print("BASE:", generate(base_model, base_tokenizer, t))
    print("LORA:", generate(lora_model, lora_tokenizer, t))
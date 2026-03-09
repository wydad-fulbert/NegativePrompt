
import gc
import time
import re
import requests
import os 


import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import os 
# ===== HF LOGIN (KAGGLE) =====
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
    login (token = token)
    
except:
    token = os.environ.get("HF_TOKEN")
    if token : 
        login(token=token)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_IDS = {
    "t5": "google/flan-t5-large",
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "vicuna": "lmsys/vicuna-7b-v1.5",
}

_MODEL_CACHE = {}
_CURRENT_MODEL_NAME = None

def get_match_items(items, str):
    match_time = 0
    str = str.lower()
    for i in items:
        i = i.strip().lower()
        if i in str:
            match_time += 1
    return match_time


def locate_ans(query, output):
    input_index = query.rfind('Input')
    input_line = query[input_index:]
    index = input_line.find('\n')
    input_line = input_line[:index]
    input_line = input_line.replace('Sentence 1:', ' ')
    input_line = input_line.replace('Sentence 2:', ' ')
    input_line = input_line.strip()
    inputs = input_line.split()
    output_lines = output.split('\n')
    # print(output_lines)
    # print('IN: ', inputs)
    ans_line = ''
    max_match_time = 0
    for i in range(len(output_lines)):
        line = output_lines[i]
        cur_match_time = get_match_items(inputs, line)
        if cur_match_time > max_match_time:
            max_match_time = cur_match_time
            ans_line = line
            if i < len(output_lines) - 1:
                ans_line += output_lines[i+1]
            if i < len(output_lines) - 2:
                ans_line += output_lines[i+2]
    # print('ANSLine: ', ans_line)
    # ans = ''
    # if len(ans_line) > 0:
    #     last_index = 0
    #     for i in inputs:
    #         i = i.strip()
    #         i_index = ans_line.rfind(i)
    #         if i_index > last_index:
    #             last_index = i_index
    #             ans = i
    # print('ANS: ', ans)
    return ans_line
    
api_num = 5

# ====== GENERATION CONFIG (PROTOCOLE) ======
MAX_NEW_TOKENS = 50
DO_SAMPLE = False          # protocole baseline (greedy)
TEMPERATURE = 0.0
TOP_P = 1.0


def clear_gpu():
    global _MODEL_CACHE
    for _, v in _MODEL_CACHE.items():
        try:
            tok, mdl, _ = v
            del tok
            del mdl
        except:
            pass
    _MODEL_CACHE = {}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(model_name):
    global _CURRENT_MODEL_NAME

# Si on change de modèle (ex: llama2 -> vicuna), on purge la VRAM
    if _CURRENT_MODEL_NAME is not None and model_name != _CURRENT_MODEL_NAME:
       clear_gpu()

    _CURRENT_MODEL_NAME = model_name

    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]


    model_id = MODEL_IDS[model_name]

    # -------- T5 --------
    if model_name == "t5":
        tokenizer = T5Tokenizer.from_pretrained(model_id)
        model = T5ForConditionalGeneration.from_pretrained(model_id).to(device)
        model.eval()
        torch.set_grad_enabled(False)
        _MODEL_CACHE[model_name] = (tokenizer, model, "t5")
        return _MODEL_CACHE[model_name]

    # -------- Llama2 / Vicuna --------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage= True
    )

    model.eval()
    if hasattr(model , "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _MODEL_CACHE[model_name] = (tokenizer, model, "causal")
    return _MODEL_CACHE[model_name]


def get_response_from_llm(llm_model, queries, task, few_shot, api_num=4):

    model_outputs = []
    tokenizer, model, model_type = load_model(llm_model)

    with torch.no_grad():
        for q in queries:

            # ===== T5 =====
            if model_type == "t5":
                inputs = tokenizer(q, return_tensors="pt").to(device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    
                )

                out_text = tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                ).strip()

            # ===== Llama2 / Vicuna =====
            else:
                inputs = tokenizer(
                    q,
                    return_tensors="pt",
                    truncation=True,
                    padding=False
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                gen_kwargs = {
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "do_sample": DO_SAMPLE,
                    "pad_token_id": tokenizer.eos_token_id,
                    "eos_token_id" : tokenizer.eos_token_id,
                    "use_cache" : True
                }

                # temperature/top_p uniquement si on sample
                if DO_SAMPLE:
                   gen_kwargs["temperature"] = TEMPERATURE
                   gen_kwargs["top_p"] = TOP_P

                outputs = model.generate(**inputs, **gen_kwargs)

                #  On enlève le prompt
                prompt_len = inputs["input_ids"].shape[1]
                gen_tokens = outputs[0][prompt_len:]

                out_text = tokenizer.decode(
                    gen_tokens, 
                    skip_special_tokens=True
                ).strip()

            model_outputs.append(out_text)

    return model_outputs

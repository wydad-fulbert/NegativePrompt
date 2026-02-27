import time
import re
import requests


import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

# ===== HF LOGIN (KAGGLE) =====
try:
    token = UserSecretsClient().get_secret("HF_TOKEN")
    if token:
        login(token=token)
except:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_IDS = {
    "t5": "google/flan-t5-large",
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "vicuna": "lmsys/vicuna-7b-v1.5",
}

_MODEL_CACHE = {}

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


def load_model(model_name):

    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    model_id = MODEL_IDS[model_name]

    # -------- T5 --------
    if model_name == "t5":
        tokenizer = T5Tokenizer.from_pretrained(model_id)
        model = T5ForConditionalGeneration.from_pretrained(model_id).to(device)
        model.eval()
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
    )

    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _MODEL_CACHE[model_name] = (tokenizer, model, "causal")
    return _MODEL_CACHE[model_name]


def get_response_from_llm(llm_model, queries, task, few_shot, api_num=4):

    model_outputs = []
    

    tokenizer, model, model_type = load_model(llm_model)
    with torch.no_grad():
        for q in queries:
            
            if model_type == "t5":
                  inputs = tokenizer(q, return_tensors="pt").to(device)
                  outputs = model.generate(
                      **inputs,
                      max_new_tokens=50,
                      do_sample=False
                  )
            else:
                  inputs = tokenizer(q, return_tensors="pt", truncation=True)
                  inputs = {k: v.to(model.device)for k, v in inputs.items()}
                  outputs = model.generate(
                      **inputs,
                      max_new_tokens=50,
                      do_sample=False,
                      pad_token_id=tokenizer.eos_token_id
                   )

            out_text = tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                ).strip()

            model_outputs.append(out_text)

    return model_outputs
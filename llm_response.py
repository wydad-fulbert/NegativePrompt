import time
import re
import requests


import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading Flan-T5 model once on {device}...")

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
model = model.to(device)
model.eval()

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

def get_response_from_llm(llm_model, queries, task, few_shot, api_num=4):

    model_outputs = []

    with torch.no_grad():
        for q in queries:
            inputs = tokenizer(q, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.0
)

            out_text = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            ).strip()

            model_outputs.append(out_text)

    return model_outputs
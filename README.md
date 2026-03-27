
# 📌 NegativePrompt — Analyse et robustesse des LLM face aux Negative Prompts



Ce dépôt contient notre projet de M2 consacré à l’étude de l’impact des *Negative Prompts* sur plusieurs modèles de langage, à partir du papier :



**“The Unreasonable Sensitivity of LLMs to Negative Prompts”**



---



## 🎯 Objectifs du projet



Ce travail s’articule autour de trois axes principaux :



- 🔁 **Reproduction** des expériences du papier sur plusieurs modèles

- 📊 **Analyse détaillée** des résultats (par tâche, métrique et intensité)

- 🚀 **Propositions d’amélioration**, notamment :

  - génération automatique de Negative Prompts

  - extension à un modèle supplémentaire (Mistral-7B-Instruct)

  - tentative de fine-tuning léger (LoRA)



---



## 🤖 Modèles étudiés



- Flan-T5-Large  

- Llama-2-7B-Chat  

- Vicuna-7B-v1.5  

- Mistral-7B-Instruct  



---



## 🧪 Tâches évaluées



### Instruction Induction

- sentiment  

- translation_en-fr  

- word_in_context  

- active_to_passive  

- negation  



### BigBench

- dyck_languages  

- object_counting  

- ruin_names  

- word_sorting  

- disambiguation_qa  



---



## ⚙️ Protocole expérimental



- Zero-shot (few-shot = False)  

- Température = 0  

- max_new_tokens = 50  

- nombre d’échantillons = 100  

- seed = 42  



### Conditions testées



- Baseline (sans Negative Prompt)  

- NP01, NP05, NP10 (intensité croissante)  



---



## 📊 Métriques d’évaluation



- **Exact Match (EM)** → tâches structurées  

- **BLEU** → tâches génératives  



---



## 📁 Structure du dépôt



```text

NegativePrompt/

├── notebooks/

│   ├── main_reproduction.ipynb

│   └── analysis.ipynb

│

├── scripts/

│   ├── reproduction/

│   ├── improvements/

│   └── utils/

│

├── results/

│   ├── reproduction/

│   └── improvements/

│

├── data/

├── README.md

├── requirements.txt

└── environment.yml

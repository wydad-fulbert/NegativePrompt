# 📌 NegativePrompt — Analyse des LLM face aux Negative Prompts

Ce dépôt contient notre projet de M2 portant sur l’étude de l’impact des *Negative Prompts* sur plusieurs modèles de langage, à partir du papier :

**“The Unreasonable Sensitivity of LLMs to Negative Prompts”**

---

##  Objectifs

Ce projet vise à :

-  Reproduire les résultats du papier 
-  Analyser l’impact des Negative Prompts 
-  Proposer des améliorations 

---

##  Modèles étudiés

- Flan-T5-Large 
- Llama-2-7B-Chat 
- Vicuna-7B-v1.5 
- Mistral-7B-Instruct 

---

##  Tâches

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

## ⚙️ Protocole

- Zero-shot 
- Température = 0 
- max_new_tokens = 50 
- 100 exemples par tâche 
- seed = 42 

### Conditions testées

- Baseline 
- NP01 
- NP05 
- NP10 

---

##  Métriques

- Exact Match (EM) 
- BLEU 

---

##  Structure du projet

```
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
```

---

##  Exécution

### Reproduction

Ouvrir et exécuter :
notebooks/main_reproduction.ipynb

---

### NP généré (amélioration principale)

python scripts/improvements/eval_final_np_t5.py

---

### Mistral

python scripts/improvements/eval_np_mistral_baseline.py

---

### LoRA

python scripts/improvements/train_t5_lora.py 
python scripts/improvements/eval_t5_lora.py 

---

## 📈 Résultats

- L’effet des Negative Prompts dépend :
  - du modèle 
  - de la tâche 
  - de l’intensité 

- Llama2 et Vicuna sont sensibles 
- Flan-T5 est plus stable 

- NP généré :
  → amélioration sur certaines tâches 

- LoRA :
  → pas d’amélioration observée 

---

##  Conclusion

La robustesse des modèles dépend fortement de la formulation du prompt. 
Adapter le prompt apparaît plus efficace que modifier le modèle.

---

##  Auteurs

- Rosine AUCLAIR 
- Besma BEKHTAOUI 
- Yasmine EL HOUSSAINI 
- Wydad FULBERT 

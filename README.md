NegativePrompt — Reproduction, analyse et pistes d’amélioration





Ce dépôt contient notre projet de M2 consacré à l’étude de l’impact des Negative Prompts sur plusieurs modèles de langage, à partir du papier “The Unreasonable Sensitivity of LLMs to Negative Prompts”.





Objectifs





Notre travail comporte trois volets :



reproduction des expériences principales du papier sur plusieurs modèles ;
analyse détaillée par tâche, par métrique et par intensité de stimulus ;
exploration de pistes d’amélioration, notamment :
génération automatique de Negative Prompts ;
extension à un modèle supplémentaire (Mistral-7B-Instruct) ;
tentative de fine-tuning léger via LoRA.







Modèles étudiés





Flan-T5-Large
Llama-2-7B-Chat
Vicuna-7B-v1.5
Mistral-7B-Instruct






Structure du dépôt





notebooks/main_reproduction.ipynb : notebook principal de reproduction
notebooks/analysis.ipynb : notebook d’analyse, tableaux et graphiques
scripts/reproduction/ : scripts de reproduction des expériences de base
scripts/improvements/ : scripts liés aux améliorations
results/reproduction/ : résultats des expériences de reproduction
results/improvements/ : résultats des expériences d’amélioration






Installation



pip install -r requirements.txt



Expériences principales







Reproduction





Exécuter le notebook :

notebooks/main_reproduction.ipynb



Amélioration principale : NP généré sur T5





Script principal :

python scripts/improvements/eval_final_np_t5.py



Extension à Mistral



python scripts/improvements/eval_np_mistral_baseline.py



Résultats principaux





Les effets des Negative Prompts dépendent fortement du modèle, de la tâche et de l’intensité.
Flan-T5-Large apparaît globalement plus stable.
La génération automatique de Negative Prompts constitue l’amélioration la plus convaincante parmi celles testées.
Le fine-tuning LoRA ne s’est pas révélé concluant dans notre cadre expérimental.






Auteurs





Rosine AUCLAIR
Besma BEKHTAOUI
Yasmine ELHOUSSAINI
Wydad FULBERT

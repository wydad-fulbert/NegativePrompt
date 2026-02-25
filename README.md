# NegativePrompt
Code release for [NegativePrompt: Leveraging Psychology for Large Language Models Enhancement via Negative Emotional Stimuli](https://arxiv.org/abs/2405.02814) (IJCAI 2024)

## Installation

First, clone the repo:
```sh
git clone git@https://github.com/wangxu0820/NegativePrompt
```

Then, 

```sh
cd NegativePrompt
```

To install the required packages, you can create a conda environment:

```sh
conda create --name negativeprompt python=3.9
```

then use pip to install required packages:

```sh
pip install -r requirements.txt
```

## Usage
```sh
python main.py --task task_name --model model_name --pnum negativeprompt_id --few_shot False
```

## Citation
Please cite us if you find this project helpful for your research:
```
@misc{wang2024negativeprompt,
      title={NegativePrompt: Leveraging Psychology for Large Language Models Enhancement via Negative Emotional Stimuli}, 
      author={Xu Wang and Cheng Li and Yi Chang and Jindong Wang and Yuan Wu},
      year={2024},
      eprint={2405.02814},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
##  Installation

### Standard installation (CPU)

```bash
pip install -r requirements.txt
```

---

## 🚀 GPU usage (Google Colab)

If you want to use GPU acceleration on Colab:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

Then restart the runtime.

You can verify GPU availability with:

```python
import torch
print(torch.cuda.is_available())
```
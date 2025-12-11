# Mistral-7B-Instruct Fine-tuning for Engineering Document Q&A

Domain adaptation of [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) for technical/engineering documentation question answering using LoRA fine-tuning.

## Overview

- **Model**: mistralai/Mistral-7B-Instruct-v0.3
- **Task**: Domain adaptation for commercial refrigeration equipment Q&A
- **Method**: LoRA (Low-Rank Adaptation) - trains only 1-2% additional parameters
- **Framework**: [mistral-finetune](https://github.com/mistralai/mistral-finetune) (official Mistral AI repository)
- **Training Data**: 266 Q&A pairs from engineering documentation (`rag_eval_QA.csv`)


## Requirements

### Hardware
- **Minimum**: NVIDIA GPU with 24GB+ VRAM (RTX 3090/4090, A5000, etc.)
- **Storage**: ~50GB for model weights and outputs

### Software
- Python 3.10+
- CUDA 11.8+ or 12.1+
- PyTorch 2.0+

> **Note**: This setup is optimized for **single GPU training (31GB VRAM)**. Multi-GPU training is supported but not required.

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/mistral-7B-instruct-finetune.git
cd mistral-7B-instruct-finetune
```

### 2. Install Dependencies

Using `uv` for fast and reliable package management:

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install mistral-finetune dependencies
cd mistral-finetune
uv venv
uv pip install -r requirements.txt

# Install additional dependencies for data preparation
cd ..
uv pip install pandas scikit-learn pyyaml jupyter ipykernel
```


### 3. Data Preparation

Your training data (`rag_eval_QA.csv`) is already in the repository. The notebook will automatically convert it to the required JSONL format.

**CSV Format:**
- `input_query`: Question text
- `output_expected_answer`: Expected answer
- `pdf_name`: Source document (optional)
- `question_number`: Question ID (optional)

### 3. Launch Jupyter Notebook

```bash
jupyter notebook mistral_7b_finetune.ipynb
```
## Advanced Usage

### Training Configuration

Default hyperparameters (optimized for 239 Q&A samples on single GPU):

```yaml
# Data
train_samples: 239 (90%)
val_samples: 27 (10%)

# Model
model: Mistral-7B-Instruct-v0.3
lora_rank: 64

# Training (Single GPU - 31GB VRAM)
seq_len: 4096        # 4K context (optimized for memory)
batch_size: 1        # Batch size per GPU
max_steps: 500       # ~2 epochs
learning_rate: 6e-5
weight_decay: 0.1
```

> **Note**: You can increase `seq_len` to 8192 or `batch_size` to 2 if you have more VRAM available.

### Command Line Training

If you prefer to skip the notebook and train directly:

**Single GPU (Recommended):**
```bash
cd mistral-finetune
./run_train.sh
```

**Multi-GPU (2+ GPUs):**
```bash
cd mistral-finetune
export CUDA_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1
torchrun --nproc_per_node=2 --master_port=29500 train.py ../train_config.yaml
```

> **Note**: For multi-GPU, you'll need to increase `batch_size` and `seq_len` in the config to utilize the additional memory.

## Project Structure

```
mistral-7B-instruct-finetune/
├── README.md                           # This file
├── rag_eval_QA.csv                     # Training data (266 Q&A pairs)
├── mistral_7b_finetune.ipynb          # Main training notebook
├── train_config.yaml                   # Generated training config
├── mistral-finetune/                   # Official mistral-finetune repo
│   ├── train.py                        # Training script
│   ├── finetune/                       # Fine-tuning modules
│   └── requirements.txt                # Dependencies
├── data/                               # Generated data (created by notebook)
│   ├── train_instruct.jsonl           # Training data (239 samples)
│   └── val_instruct.jsonl             # Validation data (27 samples)
├── models/                             # Downloaded models
│   └── mistral-7B-Instruct-v0.3/      # Base model
└── output/                             # Training outputs
    └── run_001/                        # Training run
        ├── checkpoints/                # Model checkpoints
        ├── args.yaml                   # Training arguments
        └── logs/                       # Training logs
```

### Inference

After training, load the fine-tuned model:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "models/mistral-7B-Instruct-v0.3",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("models/mistral-7B-Instruct-v0.3")

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "output/run_001/checkpoints/checkpoint_500"
)

# Generate answer
question = "What is the maximum defrost duration for an Ascend Freezer?"
prompt = f"<s>[INST] {question} [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)
```

## Output Files

After training completes:
- **LoRA Adapters**: `output/run_001/checkpoints/checkpoint_500/` (~200MB)
- **Training Logs**: `output/run_001/logs/`
- **Config**: `output/run_001/args.yaml`
- **Validation Results**: `output/validation_results.csv` (if evaluated)
- **Merged Model**: `output/merged_model/` (~15GB, if exported)

## Performance Notes

### Model Size
- **Base Model**: ~14GB (FP16)
- **LoRA Adapters**: ~200MB (rank 64)
- **Merged Model**: ~14GB

## References

- [Mistral-7B Model](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [mistral-finetune Repository](https://github.com/mistralai/mistral-finetune)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Mistral AI Documentation](https://docs.mistral.ai/)

## License

This project uses:
- Mistral-7B-Instruct-v0.3: Apache 2.0 License
- mistral-finetune: Apache 2.0 License

## Citation

```bibtex
@article{jiang2023mistral,
  title={Mistral 7B},
  author={Jiang, Albert Q and Sablayrolles, Alexandre and Mensch, Arthur and Bamford, Chris and Chaplot, Devendra Singh and Casas, Diego de las and Bressand, Florian and Lengyel, Gianna and Lample, Guillaume and Saulnier, Lucile and others},
  journal={arXiv preprint arXiv:2310.06825},
  year={2023}
}
```

## Support

For issues related to:
- **mistral-finetune**: https://github.com/mistralai/mistral-finetune/issues
- **This project**: Open an issue in this repository

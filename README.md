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
- **Recommended**: 2x NVIDIA RTX 5000 Ada (33GB each) or equivalent
- **Minimum**: NVIDIA GPU with 16GB+ VRAM (RTX 3090/4090, V100, etc.)
- **Storage**: ~50GB for model weights and outputs

### Software
- Python 3.10+
- CUDA 11.8+ or 12.1+
- PyTorch 2.0+

> **Note**: This setup is optimized for **2x RTX 5000 Ada GPUs (33GB VRAM each)**.

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

Then execute all cells sequentially. The notebook handles:
1. **Environment setup** - Create directories and paths
2. **Data preparation** - Convert CSV → JSONL format (239 train / 27 val samples)
3. **Model download** - Download Mistral-7B-Instruct-v0.3 (~14GB)
4. **Training config** - Generate optimized LoRA configuration
5. **Fine-tuning** - Train with mistral-finetune (1-3 hours depending on GPU)
6. **Evaluation** - Test on validation set
7. **Export** - Merge and save final model

## Advanced Usage

### Training Configuration

Default hyperparameters (optimized for 266 Q&A samples on 2x RTX 5000 Ada):

```yaml
# Data
train_samples: 239 (90%)
val_samples: 27 (10%)

# Model
model: Mistral-7B-Instruct-v0.3
lora_rank: 64

# Training (2x RTX 5000 Ada - 33GB each)
seq_len: 16384       # 16K context for longer engineering documents
batch_size: 8        # Total across 2 GPUs (4 per GPU)
max_steps: 500       # ~2 epochs
learning_rate: 6e-5
weight_decay: 0.1
```

**Memory usage per GPU**: ~20-24GB VRAM with these settings (out of 33GB available).

### Command Line Training

If you prefer to skip the notebook and train directly:

**Recommended - Dual GPU (2x RTX 5000 Ada - 33GB):**
```bash
cd mistral-finetune
torchrun --nproc_per_node=2 train.py ../train_config.yaml
```

**Single GPU (reduce batch_size to 4 in config):**
```bash
cd mistral-finetune
torchrun --nproc_per_node=1 train.py ../train_config.yaml
```

**4+ GPUs:**
```bash
cd mistral-finetune
torchrun --nproc_per_node=4 train.py ../train_config.yaml  # Adjust count
```

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

## Monitoring Training

### Training Logs
Monitor training progress in real-time:
```bash
tail -f output/run_001/logs/train.log
```

### Weights & Biases (Optional)
Enable W&B logging by updating `train_config.yaml`:
```yaml
wandb:
  project: "mistral-7b-engineering-qa"
  run_name: "engineering-docs-lora"
  key: "your-wandb-api-key"
  offline: False
```

## Hyperparameter Tuning

### Memory Optimization
If you encounter OOM errors on RTX 5000 Ada (33GB):
- Reduce `batch_size` (8 → 4, or 4 → 2 per GPU)
- Reduce `seq_len` (16384 → 8192)
- Reduce `lora_rank` (64 → 32)
- Use gradient checkpointing (automatically enabled in mistral-finetune)

**Note**: With 33GB VRAM, you have plenty of headroom. You could even increase to:
- `seq_len: 32768` (full 32K context) with `batch_size: 4`
- `lora_rank: 128` for better adaptation
- `batch_size: 12` (6 per GPU) for faster training

### Performance Tuning
For better adaptation:
- Increase `max_steps` (500 → 1000)
- Adjust `learning_rate` (try 3e-5 or 1e-4)
- Increase `lora_rank` (64 → 128)
- Add more training data

## Output Files

After training completes:
- **LoRA Adapters**: `output/run_001/checkpoints/checkpoint_500/` (~200MB)
- **Training Logs**: `output/run_001/logs/`
- **Config**: `output/run_001/args.yaml`
- **Validation Results**: `output/validation_results.csv` (if evaluated)
- **Merged Model**: `output/merged_model/` (~15GB, if exported)

## Performance Notes

### Training Time (500 steps)
- **2x RTX 5000 Ada (33GB)**: ~1-1.5 hours ⭐ (Your setup - 16K context, batch 8)
- **Single RTX 5000 Ada (33GB)**: ~2-2.5 hours
- **2x RTX 5000 Ada (16GB)**: ~1.5-2 hours (8K context, batch 4)
- **Single A100 (80GB)**: ~1-2 hours
- **Single RTX 4090**: ~2-3 hours

### Model Size
- **Base Model**: ~14GB (FP16)
- **LoRA Adapters**: ~200MB (rank 64)
- **Merged Model**: ~14GB

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce memory footprint
batch_size: 1
seq_len: 4096
lora_rank: 32
```

### NumPy Compatibility Error
If you see "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x":
```bash
cd mistral-finetune
uv pip install "numpy<2"
```

### Missing Dependencies
```bash
cd mistral-finetune
uv pip install --upgrade mistral-common transformers torch accelerate peft
```

### Import Errors (Already Fixed)
The repository includes fixes for `InstructTokenizerBase` import errors with mistral-common>=1.3.1.
If you encounter issues, verify imports use:
```python
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerBase
```

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

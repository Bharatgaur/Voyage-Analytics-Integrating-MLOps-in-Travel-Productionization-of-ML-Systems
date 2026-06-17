"""
src/fine_tuning.py
==================
Fine-Tuning Module — EduBot LLM Project
Industry: Education and Training

Fine-tunes a pre-trained LLM (TinyLlama / DistilGPT2 / GPT-2) on education Q&A data
using PEFT (LoRA) for parameter-efficient fine-tuning on a T4 GPU.

Designed for Google Colab T4 GPU — fits within memory limits.
Training capped at 25 epochs as per project requirements.

Model Options (choose based on available GPU RAM):
  A. TinyLlama-1.1B-Chat     — Best quality, needs ~6GB VRAM (recommended for T4)
  B. microsoft/DialoGPT-medium — Dialogue-tuned, needs ~2GB VRAM
  C. distilgpt2              — Smallest, CPU-friendly fallback

Author: EduBot Project
"""

import os
import sys
import json
import logging
import warnings
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent


# ==============================================================================
# CONFIGURATION DATACLASS
# ==============================================================================

@dataclass
class TrainingConfig:
    """
    Complete training configuration for EduBot fine-tuning.
    Adjust model_name and training parameters based on available hardware.
    """

    # ── Model ──────────────────────────────────────────────────────────────
    # Options: "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (T4 GPU, best quality)
    #          "microsoft/DialoGPT-medium"           (any GPU, moderate quality)
    #          "distilgpt2"                          (CPU fallback, fast training)
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir: str = str(ROOT_DIR / "models" / "edubot_finetuned")

    # ── LoRA Parameters ───────────────────────────────────────────────────
    use_lora: bool = True
    lora_r: int = 16           # LoRA rank — higher = more parameters trained
    lora_alpha: int = 32       # LoRA alpha — scaling factor
    lora_dropout: float = 0.05 # LoRA dropout
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj"  # attention layers for LLaMA models
    ])

    # ── Training Hyperparameters ───────────────────────────────────────────
    num_train_epochs: int = 3          # Start with 3; go up to 25 on full run
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4   # Effective batch = 4 × 4 = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 0.3
    lr_scheduler_type: str = "cosine"

    # ── Sequence ───────────────────────────────────────────────────────────
    max_seq_length: int = 512

    # ── Quantization (for T4 GPU memory efficiency) ────────────────────────
    load_in_4bit: bool = True   # 4-bit quantization via bitsandbytes
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"

    # ── Logging & Saving ───────────────────────────────────────────────────
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    save_total_limit: int = 2
    evaluation_strategy: str = "steps"
    load_best_model_at_end: bool = True

    # ── Misc ───────────────────────────────────────────────────────────────
    seed: int = 42
    fp16: bool = True                  # Use mixed precision on GPU
    dataloader_num_workers: int = 2
    report_to: str = "none"            # Set to "wandb" if you want W&B tracking
    push_to_hub: bool = False
    hub_model_id: str = "your-username/edubot-education-llm"


# ==============================================================================
# DEVICE DETECTION
# ==============================================================================

def get_device_info() -> dict:
    """Detect available hardware and return configuration."""
    info = {"cuda_available": torch.cuda.is_available(), "device": "cpu"}

    if torch.cuda.is_available():
        info["device"]        = "cuda"
        info["gpu_name"]      = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        info["cuda_version"]  = torch.version.cuda
    else:
        info["gpu_name"]      = "None"
        info["gpu_memory_gb"] = 0

    logger.info("Hardware: %s | GPU: %s | Memory: %.1f GB",
                info["device"], info["gpu_name"], info["gpu_memory_gb"])
    return info


# ==============================================================================
# DATA LOADING FOR HUGGING FACE DATASETS
# ==============================================================================

def load_hf_dataset(train_path: str, val_path: str):
    """
    Load processed JSONL/JSON data as a HuggingFace Dataset.

    Parameters
    ----------
    train_path : path to training JSON
    val_path   : path to validation JSON

    Returns
    -------
    DatasetDict with 'train' and 'validation' splits
    """
    from datasets import Dataset, DatasetDict

    train_df = pd.read_json(train_path, orient="records")
    val_df   = pd.read_json(val_path,   orient="records")

    # Keep only 'text' column for SFTTrainer
    train_dataset = Dataset.from_pandas(train_df[["text"]].reset_index(drop=True))
    val_dataset   = Dataset.from_pandas(val_df[["text"]].reset_index(drop=True))

    dataset_dict = DatasetDict({"train": train_dataset, "validation": val_dataset})

    logger.info("Dataset loaded: Train=%d | Val=%d", len(train_dataset), len(val_dataset))
    return dataset_dict


# ==============================================================================
# MODEL LOADING WITH QUANTIZATION
# ==============================================================================

def load_base_model(config: TrainingConfig):
    """
    Load the base pre-trained model with optional 4-bit quantization.

    Uses BitsAndBytes for 4-bit quantization to fit large models in T4 GPU (16GB VRAM).
    Falls back to standard float16 loading if bitsandbytes is unavailable.

    Returns
    -------
    model, tokenizer
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info("Loading base model: %s …", config.model_name)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right",   # Left-pad for generation, right-pad for training
    )

    # Ensure pad token is set (GPT-style models don't always have one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token     = tokenizer.eos_token
        tokenizer.pad_token_id  = tokenizer.eos_token_id

    # ── Quantization config ───────────────────────────────────────────────────
    if config.load_in_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,   # Nested quantization for extra savings
        )
        logger.info("Using 4-bit quantization (QLoRA mode)")
    else:
        bnb_config = None
        logger.info("Running without quantization (CPU/small GPU mode)")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    # Disable cache for gradient checkpointing compatibility
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %.2fM parameters", n_params / 1e6)

    return model, tokenizer


# ==============================================================================
# LORA SETUP
# ==============================================================================

def apply_lora(model, config: TrainingConfig):
    """
    Apply LoRA (Low-Rank Adaptation) to the model for parameter-efficient fine-tuning.

    LoRA freezes the original model weights and injects trainable low-rank
    matrices into the attention layers. This reduces trainable parameters
    by 99%+ while achieving comparable performance to full fine-tuning.

    Parameters
    ----------
    model  : pre-loaded base model
    config : TrainingConfig

    Returns
    -------
    PEFT model with LoRA adapters
    """
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    logger.info("Applying LoRA (r=%d, alpha=%d) …", config.lora_r, config.lora_alpha)

    # Prepare for k-bit training (handles gradient checkpointing with quantized model)
    if config.load_in_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameter stats
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info("Trainable parameters: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)

    return model


# ==============================================================================
# TRAINING
# ==============================================================================

def train(config: TrainingConfig, data_dir: Optional[str] = None):
    """
    Full fine-tuning pipeline:
    1. Load dataset
    2. Load base model (with quantization)
    3. Apply LoRA
    4. Train with SFTTrainer (Supervised Fine-Tuning)
    5. Save final model

    Parameters
    ----------
    config   : TrainingConfig instance
    data_dir : path to processed data directory (defaults to project data/processed/)
    """
    from transformers import TrainingArguments, EarlyStoppingCallback
    from trl import SFTTrainer

    if data_dir is None:
        data_dir = str(ROOT_DIR / "data" / "processed")

    device_info = get_device_info()

    # Adjust config for CPU (for testing without GPU)
    if not torch.cuda.is_available():
        logger.warning("No GPU detected! Switching to CPU-compatible config …")
        config.model_name              = "distilgpt2"
        config.load_in_4bit            = False
        config.fp16                    = False
        config.use_lora                = False
        config.per_device_train_batch_size = 2
        config.num_train_epochs        = 2

    # ── Load data ──────────────────────────────────────────────────────────────
    train_path = os.path.join(data_dir, "education_train.json")
    val_path   = os.path.join(data_dir, "education_val.json")
    dataset    = load_hf_dataset(train_path, val_path)

    # ── Load model ──────────────────────────────────────────────────────────────
    model, tokenizer = load_base_model(config)

    # ── Apply LoRA ──────────────────────────────────────────────────────────────
    if config.use_lora:
        model = apply_lora(model, config)

    # ── Training arguments ──────────────────────────────────────────────────────
    os.makedirs(config.output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir                  = config.output_dir,
        num_train_epochs            = config.num_train_epochs,
        per_device_train_batch_size = config.per_device_train_batch_size,
        per_device_eval_batch_size  = config.per_device_eval_batch_size,
        gradient_accumulation_steps = config.gradient_accumulation_steps,
        learning_rate               = config.learning_rate,
        weight_decay                = config.weight_decay,
        warmup_ratio                = config.warmup_ratio,
        max_grad_norm               = config.max_grad_norm,
        lr_scheduler_type           = config.lr_scheduler_type,
        logging_dir                 = os.path.join(config.output_dir, "logs"),
        logging_steps               = config.logging_steps,
        eval_strategy               = config.evaluation_strategy,
        eval_steps                  = config.eval_steps,
        save_strategy               = "steps",
        save_steps                  = config.save_steps,
        save_total_limit            = config.save_total_limit,
        load_best_model_at_end      = config.load_best_model_at_end,
        fp16                        = config.fp16 and torch.cuda.is_available(),
        seed                        = config.seed,
        dataloader_num_workers      = config.dataloader_num_workers,
        report_to                   = config.report_to,
        push_to_hub                 = config.push_to_hub,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
    )

    # ── SFTTrainer ──────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model             = model,
        args              = training_args,
        train_dataset     = dataset["train"],
        eval_dataset      = dataset["validation"],
        tokenizer         = tokenizer,
        dataset_text_field= "text",
        max_seq_length    = config.max_seq_length,
        packing           = False,   # Set True to pack short sequences for efficiency
        callbacks         = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    logger.info("=" * 60)
    logger.info("  Starting Training")
    logger.info("  Model          : %s", config.model_name)
    logger.info("  Epochs         : %d", config.num_train_epochs)
    logger.info("  Train examples : %d", len(dataset["train"]))
    logger.info("  Val examples   : %d", len(dataset["validation"]))
    logger.info("  Batch size     : %d (effective: %d)",
                config.per_device_train_batch_size,
                config.per_device_train_batch_size * config.gradient_accumulation_steps)
    logger.info("  Learning rate  : %g", config.learning_rate)
    logger.info("  LoRA           : %s (r=%d)", config.use_lora, config.lora_r)
    logger.info("=" * 60)

    # ── Train ──────────────────────────────────────────────────────────────────
    train_result = trainer.train()

    # ── Save ───────────────────────────────────────────────────────────────────
    logger.info("Saving fine-tuned model …")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # Save training metrics
    metrics_path = os.path.join(config.output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    logger.info("=" * 60)
    logger.info("  ✅ Training Complete!")
    logger.info("  Model saved to: %s", config.output_dir)
    logger.info("  Train loss    : %.4f", train_result.metrics.get("train_loss", 0))
    logger.info("=" * 60)

    return trainer, train_result


# ==============================================================================
# EVALUATION HELPER
# ==============================================================================

def compute_perplexity(trainer) -> float:
    """
    Compute perplexity on the validation set.
    Perplexity = exp(eval_loss) — lower is better.
    """
    eval_results = trainer.evaluate()
    eval_loss    = eval_results["eval_loss"]
    perplexity   = math.exp(eval_loss)
    logger.info("Validation Loss: %.4f | Perplexity: %.2f", eval_loss, perplexity)
    return perplexity


# ==============================================================================
# COLAB CONVENIENCE FUNCTIONS
# ==============================================================================

def setup_colab_environment():
    """
    Install required packages in Google Colab.
    Run this cell at the start of your Colab notebook.
    """
    install_commands = [
        "pip install -q transformers==4.40.1",
        "pip install -q datasets==2.19.0",
        "pip install -q peft==0.10.0",
        "pip install -q trl==0.8.6",
        "pip install -q bitsandbytes==0.43.1",
        "pip install -q accelerate==0.29.3",
        "pip install -q rouge-score sacrebleu evaluate",
        "pip install -q streamlit",
    ]
    print("Run these commands in a Colab cell to set up:")
    for cmd in install_commands:
        print(f"  !{cmd}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  EduBot Fine-Tuning Pipeline")
    print("  Industry: Education and Training")
    print("=" * 60)

    # Check hardware
    device_info = get_device_info()

    # Configure training
    config = TrainingConfig(
        num_train_epochs=3,   # Increase to up to 25 for full training
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )

    # Run preprocessing first if processed data doesn't exist
    train_path = ROOT_DIR / "data" / "processed" / "education_train.json"
    if not train_path.exists():
        logger.info("Processed data not found. Running preprocessing pipeline …")
        sys.path.insert(0, str(Path(__file__).parent))
        from data_collection import collect_all_data, save_raw_data
        from preprocessing   import run_full_preprocessing
        df_raw   = collect_all_data(scrape_web=False)
        raw_path = save_raw_data(df_raw)
        run_full_preprocessing(raw_path)

    # Fine-tune
    trainer, results = train(config)
    perplexity = compute_perplexity(trainer)

    print(f"\n✅ Fine-tuning complete!")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Model saved to: {config.output_dir}")

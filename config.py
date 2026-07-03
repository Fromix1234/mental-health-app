from dataclasses import dataclass, field
import torch


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    n_embd: int = 256
    n_head: int = 8
    n_layer: int = 6
    block_size: int = 192
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 4
    grad_accum_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 200
    max_steps: int = 5000
    log_interval: int = 50
    save_interval: int = 500
    eval_interval: int = 250


@dataclass
class DataConfig:
    dataset_size: int = 5000
    train_split: float = 0.9
    data_file: str = "data/therapy_data.json"
    tokenizer_file: str = "data/tokenizer.json"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = "checkpoints"
    log_file: str = "training.log"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


CONFIG = Config()
print(f"Device: {CONFIG.device}")
print(f"Model params: {CONFIG.model}")

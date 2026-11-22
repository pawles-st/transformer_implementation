#!/usr/bin/env python
import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime

import torch

from classifier import Classifier
from dataset_loader import load_reviews, load_sst2
from lstm_classifier import LSTMClassifier


RESULTS_FILE = "vram_sweep_results.csv"

MODEL_TYPES = ("transformer", "lstm")
TOKENIZER_TYPES = ("bpe", "gpt2")

DEFAULT_CONFIG = {
    "tokenizer": "bpe",
    "num_layers": 2,
    "d_e": 64,
    "d_ff": 128,       # Transformer only
    "heads": 4,        # Transformer only
    "hidden_dim": 64,  # LSTM only
    "batch_size": 16,
    "text_repeat": 1,
}

# Each experiment changes one parameter; all other values stay at baseline.
SWEEP = {
    "num_layers": (1, 2, 4, 6),
    "d_e": (32, 64, 128, 256),
    "d_ff": (64, 128, 256, 512),
    "heads": (1, 2, 4, 8),
    "hidden_dim": (32, 64, 128, 256),
    "batch_size": (16, 64, 128, 256),
    "text_repeat": (1, 4, 16, 32),
    "tokenizer": TOKENIZER_TYPES,
}

TRANSFORMER_ONLY = ("d_ff", "heads")
LSTM_ONLY = ("hidden_dim",)


def get_device() -> tuple[torch.device, str]:
    """Select the CUDA device if available, otherwise the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda"), "NVIDIA CUDA"

    return torch.device("cpu"), "CPU"


def format_memory_to_mib(memory_bytes: int | float) -> float:
    """Convert bytes to mebibytes (MiB)."""
    return memory_bytes / (1024 ** 2)


def save_results(rows: list[dict], path: str = RESULTS_FILE) -> None:
    """Append measurement rows to a CSV file."""
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults appended to {path}")


def build_model(model_type: str, config: dict) -> Classifier | LSTMClassifier:
    """Create a model using the supplied configuration."""
    if model_type == "lstm":
        return LSTMClassifier(
            num_layers=config["num_layers"],
            d_e=config["d_e"],
            hidden_dim=config["hidden_dim"],
            bidirectional=True,
            lr=1e-3,
            tokenizer_type=config["tokenizer"],
        )

    d_k = config["d_e"] // config["heads"]

    return Classifier(
        num_layers=config["num_layers"],
        d_e=config["d_e"],
        d_k=d_k,
        d_v=d_k,
        d_ff=config["d_ff"],
        h=config["heads"],
        lr=1e-3,
        tokenizer_type=config["tokenizer"],
    )


def run_vram_measurement(model_type: str, config: dict) -> None:
    """Measure allocated GPU memory during inference and one training step."""
    tokenizer_type = config["tokenizer"]
    device, backend_name = get_device()

    print(f"Device: {backend_name} ({device})")

    if device.type == "cpu":
        print("Running on CPU - VRAM statistics are unavailable.")
        return

    if tokenizer_type == "bpe":
        load_sst2(max_train_samples=1, max_val_samples=1)

    dataset = load_reviews()
    indices = [i % len(dataset) for i in range(config["batch_size"])]
    texts = [
        " ".join([dataset[i][0]] * config["text_repeat"])
        for i in indices
    ]
    labels = [dataset[i][1] for i in indices]
    model = build_model(model_type, config).to(device)
    model.optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
    )

    encoded_texts, _ = model.tokeniser.encode(texts)

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "model": model_type,
        "tokenizer": tokenizer_type,
        "num_layers": config["num_layers"],
        "d_e": config["d_e"],
        "d_ff": config["d_ff"] if model_type == "transformer" else "",
        "heads": config["heads"] if model_type == "transformer" else "",
        "hidden_dim": config["hidden_dim"] if model_type == "lstm" else "",
        "text_repeat": config["text_repeat"],
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "batch_size": len(texts),
        "unique_reviews": len(set(indices)),
        "seq_len": encoded_texts.shape[1],
    }

    print(
        f"Model: {model_type} | tokenizer: {tokenizer_type} "
        f"| layers: {meta['num_layers']} | d_e: {meta['d_e']} "
        f"| d_ff: {meta['d_ff']} | heads: {meta['heads']} "
        f"| hidden_dim: {meta['hidden_dim']} "
        f"| params: {meta['params']:,} "
        f"| batch: {meta['batch_size']} "
        f"| unique reviews: {meta['unique_reviews']} "
        f"| seq_len: {meta['seq_len']}"
    )

    model.predict(texts)
    torch.cuda.synchronize(device)

    allocated_before_inf = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    model.predict(texts)
    torch.cuda.synchronize(device)

    allocated_after_inf = torch.cuda.memory_allocated(device)
    peak_inf = torch.cuda.max_memory_allocated(device)

    inference = {
        "stage": "inference",
        "base_MiB": round(format_memory_to_mib(allocated_before_inf), 2),
        "post_MiB": round(format_memory_to_mib(allocated_after_inf), 2),
        "peak_MiB": round(format_memory_to_mib(peak_inf), 2),
        "peak_delta_MiB": round(
            format_memory_to_mib(peak_inf - allocated_before_inf), 2
        ),
    }

    print("\n--- Inference (predict) ---")
    print(f"Base Allocated: {inference['base_MiB']:.2f} MiB")
    print(f"Post Allocated: {inference['post_MiB']:.2f} MiB")
    print(f"Peak Allocated: {inference['peak_MiB']:.2f} MiB")
    print(f"Peak Delta:     {inference['peak_delta_MiB']:.2f} MiB")

    model.fit(texts, labels)
    torch.cuda.synchronize(device)

    allocated_before_train = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    model.fit(texts, labels)
    torch.cuda.synchronize(device)

    allocated_after_train = torch.cuda.memory_allocated(device)
    peak_train = torch.cuda.max_memory_allocated(device)

    training = {
        "stage": "training",
        "base_MiB": round(format_memory_to_mib(allocated_before_train), 2),
        "post_MiB": round(format_memory_to_mib(allocated_after_train), 2),
        "peak_MiB": round(format_memory_to_mib(peak_train), 2),
        "peak_delta_MiB": round(
            format_memory_to_mib(peak_train - allocated_before_train), 2
        ),
    }

    print("\n--- Training Step (fit) ---")
    print(f"Base Allocated: {training['base_MiB']:.2f} MiB")
    print(f"Post Allocated: {training['post_MiB']:.2f} MiB")
    print(f"Peak Allocated: {training['peak_MiB']:.2f} MiB")
    print(f"Peak Delta:     {training['peak_delta_MiB']:.2f} MiB")

    save_results([
        {**meta, **inference},
        {**meta, **training},
    ])


def run_in_new_process(model_type: str, config: dict) -> None:
    """Run one measurement in a separate Python process."""
    command = [sys.executable, os.path.abspath(__file__), "--model", model_type,]

    for key, value in config.items():
        command.extend([f"--{key.replace('_', '-')}", str(value)])

    subprocess.run(command, check=True)


def build_sweep_configs(model_type: str) -> list[dict]:
    """Build configurations that change one parameter at a time."""
    configs = [dict(DEFAULT_CONFIG)]

    for key, values in SWEEP.items():
        if model_type == "lstm" and key in TRANSFORMER_ONLY:
            continue

        if model_type == "transformer" and key in LSTM_ONLY:
            continue

        for value in values:
            if value != DEFAULT_CONFIG[key]:
                configs.append({**DEFAULT_CONFIG, key: value})

    return configs


def run_sweep() -> None:
    """Measure both models for all configurations in the sweep."""
    if not torch.cuda.is_available():
        print("CUDA is unavailable. VRAM measurements cannot be performed.")
        return

    for model_type in MODEL_TYPES:
        configs = build_sweep_configs(model_type)

        for number, config in enumerate(configs, start=1):
            print(
                f"\n{'=' * 50}\n"
                f"{model_type}: configuration {number}/{len(configs)}\n"
                f"{config}\n"
                f"{'=' * 50}"
            )
            run_in_new_process(model_type, config)


def main() -> None:
    """Read command-line arguments and start a sweep or one measurement."""
    parser = argparse.ArgumentParser(
        description="Measure GPU memory usage of Transformer and LSTM models."
    )
    parser.add_argument(
        "--model",
        choices=MODEL_TYPES,
        default=None,
        help="Measure only the selected model. Without this option, run the sweep.",
    )
    parser.add_argument(
        "--tokenizer",
        choices=TOKENIZER_TYPES,
        default=DEFAULT_CONFIG["tokenizer"],
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=DEFAULT_CONFIG["num_layers"],
    )
    parser.add_argument(
        "--d-e",
        type=int,
        default=DEFAULT_CONFIG["d_e"],
        help="Embedding dimension.",
    )
    parser.add_argument(
        "--d-ff",
        type=int,
        default=DEFAULT_CONFIG["d_ff"],
        help="Transformer feed-forward dimension.",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=DEFAULT_CONFIG["heads"],
        help="Number of Transformer attention heads.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=DEFAULT_CONFIG["hidden_dim"],
        help="LSTM hidden dimension per direction.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CONFIG["batch_size"],
    )
    parser.add_argument(
        "--text-repeat",
        type=int,
        default=DEFAULT_CONFIG["text_repeat"],
        help="Number of times each review is repeated to increase sequence length.",
    )

    args = parser.parse_args()

    config = {
        key: getattr(args, key)
        for key in DEFAULT_CONFIG
    }

    if args.model is None:
        run_sweep()
    else:
        run_vram_measurement(args.model, config)


if __name__ == "__main__":
    main()

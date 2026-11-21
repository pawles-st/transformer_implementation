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

RESULTS_FILE = "vram_results.csv"
MODEL_TYPES = ("transformer", "lstm")
TOKENIZER_TYPES = ("bpe", "gpt2")


def get_device() -> tuple[torch.device, str]:
    """Select the CUDA or CPU device."""
    if torch.cuda.is_available():
        return torch.device("cuda"), "NVIDIA CUDA"

    return torch.device("cpu"), "CPU"


def format_memory_to_mib(memory_bytes: int | float) -> float:
    """Convert a memory quantity from raw bytes to mebibytes (MiB)."""
    return memory_bytes / (1024**2)


def save_results(rows: list[dict], path: str = RESULTS_FILE) -> None:
    """Append measurement rows to a CSV file."""
    write_header = not os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults appended to {path}")


def build_model(model_type: str, tokenizer_type: str) -> Classifier | LSTMClassifier:
    """Create a model with default hyperparameters."""
    if model_type == "lstm":
        return LSTMClassifier(num_layers=2, d_e=64, hidden_dim=64, bidirectional=True, lr=1e-3, tokenizer_type=tokenizer_type,)

    return Classifier(num_layers=2, d_e=64, d_k=16, d_v=16, d_ff=128, h=4, lr=1e-3, tokenizer_type=tokenizer_type,)


def run_vram_measurement(model_type: str = "transformer", tokenizer_type: str = "bpe") -> None:
    """Measure VRAM allocation during inference and training."""
    
    device, backend_name = get_device()
    print(f"Device: {backend_name} ({device})")

    if device.type == "cpu":
        print("Running on CPU – VRAM statistics are unavailable.")
        return

    if tokenizer_type == "bpe":
        load_sst2(max_train_samples=1, max_val_samples=1)

    dataset = load_reviews()
    texts = [text for text, _ in dataset]
    labels = [label for _, label in dataset]

    model = build_model(model_type, tokenizer_type).to(device)
    model.optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
    )

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "model": model_type,
        "tokenizer": tokenizer_type,
        "params": sum(p.numel() for p in model.parameters()),
        "batch_size": len(texts),
        "seq_len": model.tokeniser.encode(texts)[0].shape[1],
    }

    print(
        f"Model: {model_type} | tokenizer: {tokenizer_type} "
        f"| params: {meta['params']:,} "
        f"| batch: {meta['batch_size']} "
        f"| seq_len: {meta['seq_len']}"
    )

    # Inference measurement
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
        "peak_delta_MiB": round(format_memory_to_mib(peak_inf - allocated_before_inf), 2),
    }

    print("\n--- Inference (predict) ---")
    print(f"Base Allocated: {inference['base_MiB']:.2f} MiB")
    print(f"Post Allocated: {inference['post_MiB']:.2f} MiB")
    print(f"Peak Allocated: {inference['peak_MiB']:.2f} MiB")
    print(f"Peak Delta:     {inference['peak_delta_MiB']:.2f} MiB")

    # Training measurement
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
        "peak_delta_MiB": round(format_memory_to_mib(peak_train - allocated_before_train), 2),
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


def run_all_models(tokenizer_type: str = "bpe") -> None:
    """Run each model in a separate process sequentially."""
    for model_type in MODEL_TYPES:
        print(f"\n{'=' * 50}")
        print(f"Starting measurement: {model_type}")
        print(f"{'=' * 50}")

        subprocess.run(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--model",
                model_type,
                "--tokenizer",
                tokenizer_type,
            ],
            check=True,
        )

        print(f"\nFinished measurement: {model_type}")


def main():
    """Parse CLI arguments and start VRAM measurements."""
    parser = argparse.ArgumentParser(description="Measure model VRAM usage.")
    parser.add_argument("--model", choices=MODEL_TYPES, default=None)
    parser.add_argument("--tokenizer", choices=TOKENIZER_TYPES, default="bpe")
    args = parser.parse_args()

    if args.model is None:
        run_all_models(args.tokenizer)
    else:
        run_vram_measurement(args.model, args.tokenizer)


if __name__ == "__main__":
    main()

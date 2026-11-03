"""Training and evaluation pipeline for the Stanford Sentiment Treebank (SST-2) dataset.

Trains the Classifier model sample-by-sample using its native fit(text, label) interface.
"""

import argparse
import csv
import io
import random
import time
import urllib.request
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from category import Category
from classifier import Classifier

SST2_URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
DEFAULT_DATA_DIR = Path(__file__).parent / "data" / "SST-2"


def ensure_sst2_dataset(data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """Download and extract SST-2 dataset if not already present."""
    if (data_dir / "train.tsv").exists() and (data_dir / "dev.tsv").exists():
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"SST-2 dataset not found at {data_dir}. Downloading from {SST2_URL}...")
    with urllib.request.urlopen(SST2_URL) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as z:
        # Extract files into the parent 'data' directory
        z.extractall(data_dir.parent)

    print(f"Dataset successfully extracted to {data_dir}.")


def load_sst2_split(file_path: Path, max_samples: int | None = None) -> list[tuple[str, Category]]:
    """Load a TSV split of the SST-2 dataset into (sentence, Category) tuples.

    Args:
        file_path: Path to the TSV file (e.g., train.tsv or dev.tsv).
        max_samples: Optional limit on the number of samples to load.

    Returns:
        List of (sentence_text, Category) tuples.
    """
    examples: list[tuple[str, Category]] = []
    label_map = {
        "0": Category.NEGATIVE,
        "1": Category.POSITIVE,
    }

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sentence = row["sentence"].strip()
            label_str = row["label"].strip()
            if label_str in label_map:
                examples.append((sentence, label_map[label_str]))
            if max_samples is not None and len(examples) >= max_samples:
                break

    return examples


def evaluate(
    model: Classifier,
    dataset: list[tuple[str, Category]],
    desc: str = "Evaluating",
) -> tuple[float, float]:
    """Evaluate accuracy and average cross-entropy loss over a split.

    Args:
        model: The Classifier instance.
        dataset: List of (sentence, Category) evaluation samples.
        desc: Progress description string.

    Returns:
        Tuple of (average_loss, accuracy_percentage).
    """
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for text, true_label in tqdm(dataset, desc=desc, leave=False):
            logits = model.forward(text)
            target = torch.tensor([model.category_to_idx[true_label]], dtype=torch.long)
            loss = F.cross_entropy(logits, target)
            total_loss += loss.item()

            pred_idx = torch.argmax(logits, dim=1).item()
            if model.categories[pred_idx] == true_label:
                correct += 1

    avg_loss = total_loss / len(dataset) if dataset else 0.0
    accuracy = (correct / len(dataset)) * 100.0 if dataset else 0.0
    return avg_loss, accuracy


def run_pipeline(
    epochs: int = 3,
    lr: float = 1e-4,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    num_layers: int = 2,
    d_e: int = 64,
    d_k: int = 16,
    d_v: int = 16,
    d_ff: int = 128,
    h: int = 4,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> Classifier:
    """Run full SST-2 training and evaluation pipeline."""
    ensure_sst2_dataset(data_dir)

    print(f"\nLoading SST-2 splits from: {data_dir}")
    train_data = load_sst2_split(data_dir / "train.tsv", max_samples=max_train_samples)
    val_data = load_sst2_split(data_dir / "dev.tsv", max_samples=max_val_samples)

    print(f"Loaded {len(train_data)} training examples.")
    print(f"Loaded {len(val_data)} validation examples.")

    # Instantiate model with specified hyperparameters
    model = Classifier(
        num_layers=num_layers,
        d_e=d_e,
        d_k=d_k,
        d_v=d_v,
        d_ff=d_ff,
        h=h,
        lr=lr,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized Classifier with {total_params:,} parameters (layers={num_layers}, d_e={d_e}, heads={h}).\n")

    # Initial zero-shot validation baseline
    init_val_loss, init_val_acc = evaluate(model, val_data, desc="Initial Validation")
    print(f"Pre-training Validation -> Loss: {init_val_loss:.4f} | Accuracy: {init_val_acc:.2f}%\n")

    # Training loop
    for epoch in range(1, epochs + 1):
        random.shuffle(train_data)
        model.train()

        start_time = time.perf_counter()
        running_loss = 0.0

        pbar = tqdm(train_data, desc=f"Epoch {epoch}/{epochs}")
        for step, (text, label) in enumerate(pbar, start=1):
            # Fit model on the individual text example
            model.fit(text, label)

            # Approximate running loss for progress reporting (without extra forward pass)
            with torch.no_grad():
                logits = model.forward(text)
                target = torch.tensor([model.category_to_idx[label]], dtype=torch.long)
                running_loss += F.cross_entropy(logits, target).item()

            if step % 20 == 0 or step == len(train_data):
                avg_running = running_loss / step
                pbar.set_postfix({"train_loss": f"{avg_running:.4f}"})

        epoch_time = time.perf_counter() - start_time
        samples_per_sec = len(train_data) / epoch_time

        # Epoch validation
        val_loss, val_acc = evaluate(model, val_data, desc=f"Validating Epoch {epoch}")
        avg_train_loss = running_loss / len(train_data)

        print(
            f"\nEpoch {epoch:02d} Summary:\n"
            f"  Train Loss:     {avg_train_loss:.4f}\n"
            f"  Val Loss:       {val_loss:.4f}\n"
            f"  Val Accuracy:   {val_acc:.2f}%\n"
            f"  Throughput:     {samples_per_sec:.1f} samples/sec ({epoch_time:.1f}s total)\n"
        )

    # Show a few sample predictions
    print("Sample Validation Predictions:")
    for text, actual_label in val_data[:5]:
        pred = model.predict(text)
        print(f"  [{'✓' if pred == actual_label else '✗'}] Text: \"{text[:70]}...\"")
        print(f"      Predicted: {pred.name} | Actual: {actual_label.name}")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Classifier on the SST-2 dataset.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional limit on number of training samples (e.g., 500 for a quick dry-run).",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optional limit on number of validation samples.",
    )
    parser.add_argument("--num-layers", type=int, default=2, help="Number of encoder layers.")
    parser.add_argument("--d-e", type=int, default=64, help="Embedding dimension.")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads.")

    args = parser.parse_args()

    run_pipeline(
        epochs=args.epochs,
        lr=args.lr,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        num_layers=args.num_layers,
        d_e=args.d_e,
        h=args.heads,
    )


if __name__ == "__main__":
    main()

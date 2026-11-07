"""Training and evaluation pipeline for the Stanford Sentiment Treebank (SST-2) dataset.

Trains the Classifier model on minibatches using its native fit(texts, labels) interface.
"""

import argparse
import copy
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from category import Category
from classifier import Classifier
from dataset_loader import load_sst2

CHECKPOINT_PATH = Path(__file__).parent / "data" / "SST-2" / "best_model.pt"

def evaluate(
    model: Classifier,
    dataset: list[tuple[str, Category]],
    batch_size: int = 32,
    desc: str = "Evaluating",
) -> tuple[float, float]:
    """Evaluate accuracy and average cross-entropy loss over a split using minibatches.

    Args:
        model: The Classifier instance.
        dataset: List of (sentence, Category) evaluation samples.
        batch_size: Minibatch size for evaluation.
        desc: Progress description string.

    Returns:
        Tuple of (average_loss, accuracy_percentage).
    """
    model.eval()
    total_loss = 0.0
    correct = 0

    batches = [
        dataset[i : i + batch_size]
        for i in range(0, len(dataset), batch_size)
    ]

    with torch.no_grad():
        for batch in tqdm(batches, desc=desc, leave=False):
            batch_texts = [text for text, _ in batch]
            batch_labels = [label for _, label in batch]

            logits = model.forward(batch_texts)
            targets = torch.tensor(
                [model.category_to_idx[lbl] for lbl in batch_labels],
                dtype=torch.long,
                device=logits.device,
            )
            loss = F.cross_entropy(logits, targets, reduction="sum")
            total_loss += loss.item()

            pred_indices = torch.argmax(logits, dim=1).tolist()
            for pred_idx, true_label in zip(pred_indices, batch_labels):
                if model.categories[pred_idx] == true_label:
                    correct += 1

    avg_loss = total_loss / len(dataset) if dataset else 0.0
    accuracy = (correct / len(dataset)) * 100.0 if dataset else 0.0
    return avg_loss, accuracy


def run_pipeline(
    epochs: int = 6,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    batch_size: int = 32,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    num_layers: int = 2,
    d_e: int = 64,
    d_k: int = 16,
    d_v: int = 16,
    d_ff: int = 128,
    h: int = 4,
) -> Classifier:
    """Run full SST-2 training and evaluation pipeline."""
    print("\nLoading SST-2 datasets...")
    train_data, val_data = load_sst2(
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )

    print(f"Loaded {len(train_data)} training examples.")
    print(f"Loaded {len(val_data)} validation examples.")

    # Instantiate model with specified hyperparameters and AdamW
    model = Classifier(
        num_layers=num_layers,
        d_e=d_e,
        d_k=d_k,
        d_v=d_v,
        d_ff=d_ff,
        h=h,
        lr=lr,
        weight_decay=weight_decay,
    )

    # Cosine annealing learning rate scheduler over training epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model.optimizer,
        T_max=epochs,
        eta_min=lr * 0.1,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Initialized Classifier with {total_params:,} parameters "
        f"(layers={num_layers}, d_e={d_e}, heads={h}, batch_size={batch_size}, "
        f"lr={lr}, weight_decay={weight_decay}).\n"
    )

    # Initial zero-shot validation baseline
    init_val_loss, init_val_acc = evaluate(model, val_data, batch_size=batch_size, desc="Initial Validation")
    print(f"Pre-training Validation -> Loss: {init_val_loss:.4f} | Accuracy: {init_val_acc:.2f}%\n")

    # Tracking for saving the best validation checkpoint
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_checkpoint_path = CHECKPOINT_PATH

    # Training loop
    for epoch in range(1, epochs + 1):
        random.shuffle(train_data)
        model.train()

        start_time = time.perf_counter()
        running_loss = 0.0
        correct_train = 0
        total_samples = 0

        # Chunk training data into minibatches
        batches = [
            train_data[i : i + batch_size]
            for i in range(0, len(train_data), batch_size)
        ]

        pbar = tqdm(batches, desc=f"Epoch {epoch}/{epochs}")
        for step, batch in enumerate(pbar, start=1):
            batch_texts = [text for text, _ in batch]
            batch_labels = [label for _, label in batch]

            # Fit model on the minibatch and retrieve batch loss & correct count
            loss, n_correct = model.fit(batch_texts, batch_labels)
            running_loss += float(loss) * len(batch)
            correct_train += n_correct
            total_samples += len(batch)

            if step % 20 == 0 or step == len(batches):
                avg_running_loss = running_loss / total_samples
                running_acc = (correct_train / total_samples) * 100.0
                pbar.set_postfix({
                    "train_loss": f"{avg_running_loss:.4f}",
                    "train_acc": f"{running_acc:.2f}%",
                })

        epoch_time = time.perf_counter() - start_time
        samples_per_sec = len(train_data) / epoch_time

        # Capture current learning rate and step scheduler
        current_lr = model.optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Epoch testing on validation split
        val_loss, val_acc = evaluate(
            model, val_data, batch_size=batch_size, desc=f"Testing Epoch {epoch} (Val)"
        )
        avg_train_loss = running_loss / total_samples if total_samples > 0 else 0.0
        train_acc = (correct_train / total_samples) * 100.0 if total_samples > 0 else 0.0
        overfitting_gap = train_acc - val_acc

        # Save checkpoint if this epoch reached the best validation accuracy
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(best_state_dict, best_checkpoint_path)

        best_marker = " ★ (NEW BEST)" if is_best else ""
        print(
            f"\nEpoch {epoch:02d} Summary (Testing & Overfitting Check):\n"
            f"  Train Loss:     {avg_train_loss:.4f}\n"
            f"  Train Accuracy: {train_acc:.2f}%\n"
            f"  Val Loss:       {val_loss:.4f}\n"
            f"  Val Accuracy:   {val_acc:.2f}%{best_marker}\n"
            f"  Overfitting Gap (Train - Val Acc): {overfitting_gap:+.2f}%\n"
            f"  Learning Rate:  {current_lr:.6e}\n"
            f"  Throughput:     {samples_per_sec:.1f} samples/sec ({epoch_time:.1f}s total)\n"
        )

    # Restore the best validation model checkpoint
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(
            f"\nRestored best model checkpoint from Epoch {best_epoch:02d}:\n"
            f"  Val Accuracy:   {best_val_acc:.2f}%\n"
            f"  Val Loss:       {best_val_loss:.4f}\n"
            f"  Checkpoint:     {best_checkpoint_path}\n"
        )

    # Show a few sample predictions using the best model
    print("Sample Validation Predictions (Best Model):")
    for text, actual_label in val_data[:10]:
        pred = model.predict(text)
        print(f"  [{'✓' if pred == actual_label else '✗'}] Text: \"{text[:70]}...\"")
        print(f"      Predicted: {pred.name} | Actual: {actual_label.name}")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Classifier on the SST-2 dataset.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Minibatch size for training and evaluation.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay penalty.")
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
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        num_layers=args.num_layers,
        d_e=args.d_e,
        h=args.heads,
    )


if __name__ == "__main__":
    main()

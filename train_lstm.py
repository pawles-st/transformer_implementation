"""Train and evaluate an LSTM sentiment classifier on SST-2."""

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from tokeniser import Tokeniser
from embedder import Embedder
from lstm import LSTMM
from train_sst2 import (
    DEFAULT_DATA_DIR,
    ensure_sst2_dataset,
    load_sst2_split,
)


def prepare_data(dataset, tokeniser, category_to_idx):
    """Tokenize once; keep token IDs and targets on CPU."""
    return [
        (
            tokeniser.encode(text),
            torch.tensor([category_to_idx[label]], dtype=torch.long),
        )
        for text, label in tqdm(dataset, desc="Tokenizing")
    ]


@torch.no_grad()
def evaluate(model, embedder, examples, device):
    """Return average cross-entropy loss and accuracy percentage."""
    model.eval()
    embedder.eval()

    total_loss = 0.0
    correct = 0

    for tokens, target in examples:
        tokens = tokens.to(device)
        target = target.to(device)

        embeddings = embedder(tokens).unsqueeze(0)
        logits = model(embeddings)

        total_loss += F.cross_entropy(logits, target).item()
        correct += (logits.argmax(dim=1) == target).sum().item()

    return total_loss / len(examples), 100.0 * correct / len(examples)


def run_pipeline(
    epochs=5,
    lr=1e-3,
    max_train_samples=200,
    max_val_samples=100,
    embedding_dim=64,
    hidden_size=64,
    num_layers=1,
    seed=42,
    data_dir=DEFAULT_DATA_DIR,
    output_path=Path("checkpoints/lstm_last.pt"),
):
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(data_dir)
    print("Device:", device)

    ensure_sst2_dataset(data_dir)

    train_data = load_sst2_split(
        data_dir / "train.tsv",
        max_samples=max_train_samples,
    )
    val_data = load_sst2_split(
        data_dir / "dev.tsv",
        max_samples=max_val_samples,
    )

    if not train_data or not val_data:
        raise ValueError("Training and validation datasets must be non-empty.")

    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")

    tokeniser = Tokeniser()
    embedder = Embedder(
        embedding_dim, tokeniser.vocab_size
    ).to(device)
    model = LSTMM(
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(device)

    category_to_idx = {
        category: i for i, category in enumerate(model.categories)
    }

    train_examples = prepare_data(train_data, tokeniser, category_to_idx)
    val_examples = prepare_data(val_data, tokeniser, category_to_idx)

    parameters = list(model.parameters()) + list(embedder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)

    print(f"Trainable parameters (including embedder): "
          f"{sum(p.numel() for p in parameters):,}")

    val_loss, val_acc = evaluate(model, embedder, val_examples, device)
    print(
        f"Before training | "
        f"val loss: {val_loss:.4f} | val accuracy: {val_acc:.1f}%"
    )

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        embedder.train()
        random.shuffle(train_examples)

        start = time.perf_counter()
        running_loss = 0.0

        progress = tqdm(train_examples, desc=f"Epoch {epoch}/{epochs}")
        for step, (tokens, target) in enumerate(progress, start=1):
            tokens = tokens.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            embeddings = embedder(tokens).unsqueeze(0)
            logits = model(embeddings)
            loss = F.cross_entropy(logits, target)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % 20 == 0 or step == len(train_examples):
                progress.set_postfix(loss=f"{running_loss / step:.4f}")

        if device.type == "cuda":
            torch.cuda.synchronize()
        training_seconds = time.perf_counter() - start

        train_loss, train_acc = evaluate(
            model, embedder, train_examples, device
        )
        val_loss, val_acc = evaluate(
            model, embedder, val_examples, device
        )

        history.append({
            "epoch": epoch,
            "online_train_loss": running_loss / len(train_examples),
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "training_seconds": training_seconds,
        })

        print(
            f"\nEpoch {epoch:02d}\n"
            f"  Train loss: {train_loss:.4f} | accuracy: {train_acc:.1f}%\n"
            f"  Val loss:   {val_loss:.4f} | accuracy: {val_acc:.1f}%\n"
            f"  Training time (excluding evaluation): "
            f"{training_seconds:.1f}s\n"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedder_state_dict": embedder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "embedding_dim": embedding_dim,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "vocab_size": tokeniser.vocab_size,
                "categories": [cat.value for cat in model.categories],
                "lr": lr,
                "seed": seed,
                "epochs": epochs,
            },
            "history": history,
        },
        output_path,
    )
    print(f"Saved final checkpoint: {output_path}")

    return model, embedder, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train-samples", type=int, default=200)
    parser.add_argument("--max-val-samples", type=int, default=100)
    parser.add_argument("--full-data", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/lstm_last.pt")
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")
    if not args.full_data and (
        args.max_train_samples < 1 or args.max_val_samples < 1
    ):
        parser.error("Sample limits must be positive.")

    run_pipeline(
        epochs=args.epochs,
        lr=args.lr,
        max_train_samples=None if args.full_data else args.max_train_samples,
        max_val_samples=None if args.full_data else args.max_val_samples,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        seed=args.seed,
        data_dir=args.data_dir,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
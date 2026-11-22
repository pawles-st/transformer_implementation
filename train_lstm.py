"""Train and evaluate an LSTM classifier on AG News (batched)."""

import argparse
import copy
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from tokeniser import BPETokeniser, GPT2Tokeniser
from embedder import Embedder
from lstm import LSTMM
from dataset_loader import load_ag_news

CHECKPOINT_PATH = Path("checkpoints/lstm_best.pt")


def make_batches(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


def forward_batch(model, embedder, tokeniser, batch, category_to_idx, device):
    texts = [text for text, _ in batch]
    targets = torch.tensor(
        [category_to_idx[label] for _, label in batch],
        dtype=torch.long, device=device,
    )
    tokens, mask = tokeniser.encode(texts)
    tokens, mask = tokens.to(device), mask.to(device)
    logits = model(embedder(tokens), mask)
    return logits, targets


@torch.no_grad()
def evaluate(model, embedder, tokeniser, data, category_to_idx, device,
             batch_size=32):
    model.eval()
    embedder.eval()
    total_loss, correct = 0.0, 0
    for batch in make_batches(data, batch_size):
        logits, targets = forward_batch(
            model, embedder, tokeniser, batch, category_to_idx, device)
        total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
        correct += (logits.argmax(dim=1) == targets).sum().item()
    return total_loss / len(data), 100.0 * correct / len(data)


def run_pipeline(
    epochs=6,
    lr=1e-3,
    weight_decay=0.01,
    batch_size=32,
    max_train_samples=None,
    max_val_samples=None,
    embedding_dim=64,
    hidden_size=64,
    num_layers=1,
    dropout=0.5,
    embedding_dropout=0.3,
    pooling="mean_max",
    tokeniser_name="bpe",
    vocab_size=3000,
    seed=42,
    device="auto",
    output_path=CHECKPOINT_PATH,
):
    random.seed(seed)
    torch.manual_seed(seed)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print("Device:", device)

    # load_ag_news downloads the data first (BPETokeniser trains on it)
    train_data, val_data = load_ag_news(
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )
    if not train_data or not val_data:
        raise ValueError("Training and validation datasets must be non-empty.")
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")

    if tokeniser_name == "gpt2":
        tokeniser = GPT2Tokeniser()
    else:
        tokeniser = BPETokeniser(vocab_size=vocab_size)

    embedder = Embedder(
        embedding_dim, tokeniser.vocab_size, padding_idx=tokeniser.pad_id
    ).to(device)
    model = LSTMM(
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_rate=dropout,
        embedding_dropout=embedding_dropout,
        pooling=pooling,
    ).to(device)
    category_to_idx = {c: i for i, c in enumerate(model.categories)}

    parameters = list(model.parameters()) + list(embedder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.1)

    print(f"Config: layers={num_layers}, hidden={hidden_size}, "
          f"dropout={dropout}, emb_dropout={embedding_dropout}, "
          f"pooling={pooling}, weight_decay={weight_decay}, seed={seed}")
    print(f"Trainable parameters (including embedder): "
          f"{sum(p.numel() for p in parameters):,}")

    val_loss, val_acc = evaluate(model, embedder, tokeniser, val_data,
                                 category_to_idx, device, batch_size)
    print(f"Before training | val loss: {val_loss:.4f} | "
          f"val accuracy: {val_acc:.1f}%")

    history = []
    best_val_acc, best_epoch, best_state = -1.0, 0, None

    for epoch in range(1, epochs + 1):
        model.train()
        embedder.train()
        random.shuffle(train_data)

        current_lr = optimizer.param_groups[0]["lr"]
        start = time.perf_counter()
        running_loss, correct_train, seen = 0.0, 0, 0

        progress = tqdm(make_batches(train_data, batch_size),
                        desc=f"Epoch {epoch}/{epochs}")
        for batch in progress:
            optimizer.zero_grad()
            logits, targets = forward_batch(
                model, embedder, tokeniser, batch, category_to_idx, device)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch)
            correct_train += (logits.argmax(dim=1) == targets).sum().item()
            seen += len(batch)
            progress.set_postfix(
                train_loss=f"{running_loss / seen:.4f}",
                train_acc=f"{100.0 * correct_train / seen:.1f}%")

        if device.type == "cuda":
            torch.cuda.synchronize()
        training_seconds = time.perf_counter() - start
        scheduler.step()

        val_loss, val_acc = evaluate(model, embedder, tokeniser, val_data,
                                     category_to_idx, device, batch_size)
        train_loss = running_loss / seen
        train_acc = 100.0 * correct_train / seen

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc, best_epoch = val_acc, epoch
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "embedder": copy.deepcopy(embedder.state_dict()),
            }

        history.append({
            "epoch": epoch, "lr": current_lr,
            "train_loss": train_loss, "train_accuracy": train_acc,
            "val_loss": val_loss, "val_accuracy": val_acc,
            "training_seconds": training_seconds,
        })
        print(
            f"\nEpoch {epoch:02d}{' (NEW BEST)' if is_best else ''}\n"
            f"  Train loss: {train_loss:.4f} | accuracy: {train_acc:.1f}%\n"
            f"  Val loss:   {val_loss:.4f} | accuracy: {val_acc:.1f}%\n"
            f"  Gap (train - val): {train_acc - val_acc:+.1f}%\n"
            f"  LR: {current_lr:.2e} | training time: {training_seconds:.1f}s\n"
        )

    model.load_state_dict(best_state["model"])
    embedder.load_state_dict(best_state["embedder"])
    print(f"Restored best model from epoch {best_epoch}: "
          f"val accuracy {best_val_acc:.1f}%")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedder_state_dict": embedder.state_dict(),
            "config": {
                "embedding_dim": embedding_dim,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "embedding_dropout": embedding_dropout,
                "pooling": pooling,
                "tokeniser": tokeniser_name,
                "vocab_size": tokeniser.vocab_size,
                "pad_id": tokeniser.pad_id,
                "categories": [c.value for c in model.categories],
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "seed": seed,
                "epochs": epochs,
                "best_epoch": best_epoch,
            },
            "history": history,
        },
        output_path,
    )
    print(f"Saved best checkpoint: {output_path}")
    return model, embedder, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--embedding-dropout", type=float, default=0.3)
    parser.add_argument("--pooling", choices=["last", "mean", "max", "mean_max"],
                        default="mean_max")
    parser.add_argument("--tokeniser", choices=["bpe", "gpt2"], default="bpe")
    parser.add_argument("--vocab-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto",
                        help="auto, cpu or cuda")
    parser.add_argument("--output", type=Path, default=CHECKPOINT_PATH)
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")

    run_pipeline(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        embedding_dropout=args.embedding_dropout,
        pooling=args.pooling,
        tokeniser_name=args.tokeniser,
        vocab_size=args.vocab_size,
        seed=args.seed,
        device=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
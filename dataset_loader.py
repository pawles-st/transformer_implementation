"""Module for loading datasets: Reviews and SST-2."""

import csv
import io
import json
import urllib.request
import zipfile
from pathlib import Path

from category import Category

REVIEWS_FILE = Path(__file__).parent / "reviews.json"
SST2_DIR = Path(__file__).parent / "data" / "SST-2"
SST2_URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"

def load_reviews() -> list[tuple[str, Category]]:
    """Load reviews from reviews.json into (text, Category) tuples."""
    with open(REVIEWS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    label_map = {
        "positive": Category.POSITIVE,
        "negative": Category.NEGATIVE,
    }
    return [(item["text"], label_map[item["label"].lower()]) for item in data]


def _ensure_sst2() -> None:
    """Download and extract SST-2 dataset if not already present."""
    if (SST2_DIR / "train.tsv").exists() and (SST2_DIR / "dev.tsv").exists():
        return

    SST2_DIR.mkdir(parents=True, exist_ok=True)
    print(f"SST-2 dataset not found at {SST2_DIR}. Downloading from {SST2_URL}...")
    with urllib.request.urlopen(SST2_URL) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as z:
        z.extractall(SST2_DIR.parent)

    print(f"Dataset successfully extracted to {SST2_DIR}.")


def _load_tsv(file_path: Path, max_samples: int | None = None) -> list[tuple[str, Category]]:
    examples: list[tuple[str, Category]] = []
    label_map = {
        "0": Category.NEGATIVE,
        "1": Category.POSITIVE,
    }
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sentence = row["sentence"].strip()
            label_str = row.get("label", "").strip()
            if label_str in label_map:
                examples.append((sentence, label_map[label_str]))
            if max_samples is not None and len(examples) >= max_samples:
                break
    return examples


def load_sst2(
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> tuple[list[tuple[str, Category]], list[tuple[str, Category]]]:
    """Load the SST-2 dataset, returning (train_data, val_data)."""
    _ensure_sst2()
    train_data = _load_tsv(SST2_DIR / "train.tsv", max_samples=max_train_samples)
    val_data = _load_tsv(SST2_DIR / "dev.tsv", max_samples=max_val_samples)
    return train_data, val_data

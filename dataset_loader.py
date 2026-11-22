"""Module for loading datasets: Reviews and AG News."""

import csv
import urllib.request
from pathlib import Path

from category import Category

AG_NEWS_DIR = Path(__file__).parent / "data" / "ag_news"
AG_NEWS_URLS = {
    "train.csv": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    "test.csv": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
}

AG_NEWS_LABEL_MAP = {
    "1": Category.WORLD,
    "2": Category.SPORTS,
    "3": Category.BUSINESS,
    "4": Category.SCI_TECH,
}


def _ensure_ag_news() -> None:
    """Download AG News dataset splits if not already present."""
    AG_NEWS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in AG_NEWS_URLS.items():
        target_path = AG_NEWS_DIR / filename
        if not target_path.exists():
            print(f"AG News split not found at {target_path}. Downloading from {url}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response:
                target_path.write_bytes(response.read())
            print(f"Downloaded {filename} to {target_path}.")


def _load_ag_news_csv(file_path: Path, max_samples: int | None = None) -> list[tuple[str, Category]]:
    """Load an AG News CSV split into (text, Category) tuples."""
    examples: list[tuple[str, Category]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            label_idx = row[0].strip()
            title = row[1].strip() if len(row) > 1 else ""
            description = row[2].strip() if len(row) > 2 else ""
            text = f"{title} {description}".strip()
            if label_idx in AG_NEWS_LABEL_MAP:
                examples.append((text, AG_NEWS_LABEL_MAP[label_idx]))
            if max_samples is not None and len(examples) >= max_samples:
                break
    return examples


def load_ag_news(
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> tuple[list[tuple[str, Category]], list[tuple[str, Category]]]:
    """Load the AG News dataset, returning (train_data, val_data)."""
    _ensure_ag_news()
    train_data = _load_ag_news_csv(AG_NEWS_DIR / "train.csv", max_samples=max_train_samples)
    val_data = _load_ag_news_csv(AG_NEWS_DIR / "test.csv", max_samples=max_val_samples)
    return train_data, val_data


# Alias for backward compatibility if any legacy scripts invoke load_sst2
load_sst2 = load_ag_news

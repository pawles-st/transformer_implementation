"""Module for reading and parsing the sentiment reviews dataset."""

import json
from pathlib import Path
from category import Category


def load_dataset(file_path: str | Path | None = None) -> list[tuple[str, Category]]:
    """Read reviews from a JSON dataset file and return (text, Category) tuples.

    Args:
        file_path: Path to the JSON dataset file. If None, defaults to 'reviews.json'
            in the same directory.

    Returns:
        A list of (review_text, category_enum) tuples.
    """
    if file_path is None:
        file_path = Path(__file__).parent / "reviews.json"
    else:
        file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    label_map = {
        "positive": Category.POSITIVE,
        "negative": Category.NEGATIVE,
    }

    return [(item["text"], label_map[item["label"].lower()]) for item in data]


if __name__ == "__main__":
    reviews = load_dataset()
    print(f"Loaded {len(reviews)} reviews.")
    pos_count = sum(1 for _, label in reviews if label == Category.POSITIVE)
    neg_count = sum(1 for _, label in reviews if label == Category.NEGATIVE)
    print(f"Positive: {pos_count}, Negative: {neg_count}")

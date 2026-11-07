"""Unit/integration test verifying that the Classifier model actively learns over time."""

import torch
import torch.nn.functional as F
from category import Category
from classifier import Classifier
from dataset_loader import load_reviews


def compute_dataset_loss(model: Classifier, dataset: list[tuple[str, Category]]) -> float:
    """Compute average cross-entropy loss over a dataset without updating weights."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for text, label in dataset:
            logits = model.forward(text)
            target = torch.tensor([model.category_to_idx[label]], dtype=torch.long)
            loss = F.cross_entropy(logits, target)
            total_loss += loss.item()
    return total_loss / len(dataset)


def compute_accuracy(model: Classifier, dataset: list[tuple[str, Category]]) -> float:
    """Compute percentage of correct predictions over a dataset."""
    model.eval()
    correct = sum(1 for text, label in dataset if model.predict(text) == label)
    return (correct / len(dataset)) * 100.0


def test_model_learns() -> None:
    """Test that Classifier reduces loss and improves accuracy when fitting data."""
    # Seed for reproducibility
    torch.manual_seed(42)

    # Use a small subset of reviews for fast, deterministic learning verification
    full_dataset = load_reviews()
    toy_data = full_dataset[:6]  # 3 positive, 3 negative examples

    # Instantiate the classifier with small dimensions for fast testing
    model = Classifier(
        num_layers=2,
        d_e=64,
        d_k=16,
        d_v=16,
        d_ff=128,
        h=4,
        lr=2e-3,
    )

    # Record initial weights of a parameter to verify weights change
    initial_param = model.head.weight.clone()

    # Measure initial metrics
    initial_loss = compute_dataset_loss(model, toy_data)
    initial_acc = compute_accuracy(model, toy_data)
    print(f"\nInitial State -> Loss: {initial_loss:.4f} | Accuracy: {initial_acc:.1f}%")

    # Train for a few epochs on the toy subset
    epochs = 10
    for epoch in range(1, epochs + 1):
        for text, label in toy_data:
            model.fit(text, label)

    # Measure final metrics
    final_loss = compute_dataset_loss(model, toy_data)
    final_acc = compute_accuracy(model, toy_data)
    print(f"Final State (Epoch {epochs}) -> Loss: {final_loss:.4f} | Accuracy: {final_acc:.1f}%")

    # Assertions:
    # 1. Weights must have changed (gradients propagated and optimizer stepped)
    assert not torch.equal(initial_param, model.head.weight), (
        "Model parameters did not change after training steps."
    )

    # 2. Loss must have significantly decreased
    assert final_loss < initial_loss * 0.5, (
        f"Loss did not decrease sufficiently: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )

    # 3. Accuracy must have reached high level (overfitting small batch)
    assert final_acc >= 80.0, (
        f"Model failed to achieve high accuracy on toy batch: {final_acc:.1f}%"
    )

    print("SUCCESS: Model successfully learns and reduces loss over time.")


if __name__ == "__main__":
    test_model_learns()

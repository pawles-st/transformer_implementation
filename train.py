"""Script for training the Classifier and testing if it achieves near-zero error."""

import random
from category import Category
from classifier import Classifier
from dataset_reader import load_dataset


def run_training(epochs: int = 50, lr: float = 1e-3) -> None:
    """Train Classifier on the dataset and evaluate whether training error reaches near-zero.

    Args:
        epochs: Number of training epochs.
        lr: Learning rate for optimizer.
    """
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} examples from dataset.")

    # Instantiate the classifier model
    model = Classifier(
        num_layers=2,
        d_e=64,
        d_k=16,
        d_v=16,
        d_ff=128,
        h=4,
        lr=lr,
    )

    print(f"Training classifier for {epochs} epochs...")
    for epoch in range(0, epochs + 1):
        random.shuffle(dataset)

        # Fit model on each review example
        for text, label in dataset:
            model.fit(text, label)

        # Monitor loss/error convergence
        if epoch % 10 == 0 or epoch == epochs:
            correct = sum(1 for text, label in dataset if model.predict(text) == label)
            accuracy = (correct / len(dataset)) * 100.0
            error = 100.0 - accuracy
            print(f"Epoch {epoch:02d} | Error: {error:5.1f}% | Accuracy: {accuracy:5.1f}% ({correct}/{len(dataset)})")

    print("\nSample predictions:")
    for text, true_label in dataset[:4]:
        pred = model.predict(text)
        print(f"Text: '{text}'")
        print(f"  -> Predicted: {pred.name}, Actual: {true_label.name}")


if __name__ == "__main__":
    run_training()

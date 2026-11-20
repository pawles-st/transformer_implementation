import csv
import os
from datetime import datetime
import torch
from classifier import Classifier
from dataset_reader import load_dataset


RESULTS_FILE = "vram_results.csv" 

def get_device() -> tuple[torch.device, str]:
    """Select the CUDA or CPU device."""
    if torch.cuda.is_available():
        return torch.device("cuda"), "NVIDIA CUDA"

    return torch.device("cpu"), "CPU"


def format_memory_to_mib(memory_bytes: int | float) -> float:
    """Convert a memory quantity from raw bytes to mebibytes (MiB).

    Args:
        memory_bytes: Quantity of memory in bytes.

    Returns:
        float: The equivalent quantity formatted in MiB (1 MiB = 1024^2 bytes).
    """
    return memory_bytes / (1024**2)


def save_results(rows: list[dict], path: str = RESULTS_FILE) -> None:
    """Append measurement rows to a CSV file.

    The header is written only when the file does not exist yet, so results of
    many runs (e.g. on different GPUs) end up in one file. If the columns of the
    script change, delete the old file first.

    Args:
        rows: Rows (dictionaries) with identical keys.
        path: Path of the CSV file (relative to the directory the script is run from).
    """
    write_header = not os.path.exists(path)  
    with open(path, "a", newline="", encoding="utf-8") as f:  
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults appended to {path}")


def run_vram_measurement() -> None:
    """Execute VRAM allocation metrics for inference and single-step training.

    Initializes the Classifier model on the GPU and reads memory statistics
    using PyTorch's CUDA allocator.

    Measurement Protocol:
        - Runs an unmeasured warm-up pass before each stage to force GPU context
          initialization and lazy tensor allocations (e.g., Adam momentum buffers).
        - Synchronizes CPU and GPU streams via torch.cuda.synchronize() to guarantee
          accurate hardware counters.
        - Resets peak memory trackers via torch.cuda.reset_peak_memory_stats() right
          before the measured call.

    Prints:
        Base allocated memory, post-execution allocated memory, peak memory, and
        peak memory deltas in MiB. Exits gracefully if executed on a CPU.

    Saves:
        The same values (one row for inference and one for training, together with
        the date, GPU name and PyTorch version) are appended to RESULTS_FILE.
        Nothing is saved on a CPU.
    """
    device, backend_name = get_device()
    print(f"Device: {backend_name} ({device})")

    if device.type == "cpu":
        print("Running on CPU – dedicated VRAM statistics are not available.")
        return

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
    }

    dataset = load_dataset()
    text, label = dataset[0]

    model = Classifier(
        num_layers=2,
        d_e=64,
        d_k=16,
        d_v=16,
        d_ff=128,
        h=4,
        lr=1e-3,
    ).to(device)

    model.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.predict(text)
    torch.cuda.synchronize(device)

    allocated_before_inf = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    model.predict(text)
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
    print(f"Peak Delta: {inference['peak_delta_MiB']:.2f} MiB")

    model.fit(text, label)
    torch.cuda.synchronize(device)

    allocated_before_train = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    model.fit(text, label)
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
    print(f"Base Allocated (with Adam states): {training['base_MiB']:.2f} MiB")
    print(f"Post Allocated: {training['post_MiB']:.2f} MiB")
    print(f"Peak Allocated: {training['peak_MiB']:.2f} MiB")
    print(f"Peak Delta: {training['peak_delta_MiB']:.2f} MiB")
    save_results([{**meta, **inference}, {**meta, **training}])


if __name__ == "__main__":
    run_vram_measurement()

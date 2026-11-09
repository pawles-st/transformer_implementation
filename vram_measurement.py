import torch
from classifier import Classifier
from dataset_reader import load_dataset


def get_device() -> tuple[torch.device, str]:
    """Select the best available hardware accelerator and determine its backend string.

    Checks hardware backends in order of availability:
        1. NVIDIA CUDA (standard CUDA backend)
        2. AMD ROCm (CUDA compatibility layer via AMD HIP)
        3. Intel XPU (Intel OneAPI acceleration layer)
        4. CPU (fallback mode)

    Returns:
        tuple[torch.device, str]: A tuple containing:
            - torch.device: Configured PyTorch device object.
            - str: Human-readable backend description ("NVIDIA CUDA", "AMD ROCm",
              "Intel XPU", or "CPU").
    """
    if torch.cuda.is_available():
        if getattr(torch.version, "hip", None) is not None:
            return torch.device("cuda"), "AMD ROCm"
        return torch.device("cuda"), "NVIDIA CUDA"

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu"), "Intel XPU"

    return torch.device("cpu"), "CPU"


def format_memory_to_mib(memory_bytes: int | float) -> float:
    """Convert a memory quantity from raw bytes to mebibytes (MiB).

    Args:
        memory_bytes: Quantity of memory in bytes.

    Returns:
        float: The equivalent quantity formatted in MiB (1 MiB = 1024^2 bytes).
    """
    return memory_bytes / (1024**2)


def run_vram_measurement() -> None:
    """Execute VRAM allocation metrics for inference and single-step training.

    Initializes the Classifier model on the target hardware accelerator and reads
    memory statistics using PyTorch's backend allocator.

    Measurement Protocol:
        - Runs an unmeasured warm-up pass before each stage to force GPU context
          initialization and lazy tensor allocations (e.g., Adam momentum buffers).
        - Synchronizes CPU and GPU streams via backend.synchronize() to guarantee
          accurate hardware counters.
        - Resets peak memory trackers via backend.reset_peak_memory_stats() right
          before the measured call.

    Prints:
        Base allocated memory, post-execution allocated memory, peak memory, and
        peak memory deltas in MiB. Exits gracefully if executed on a CPU.
    """
    device, backend_name = get_device()
    print(f"Device: {backend_name} ({device})")

    if device.type == "cpu":
        print("Running on CPU – dedicated VRAM statistics are not available.")
        return

    backend = torch.cuda if device.type == "cuda" else torch.xpu

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
    backend.synchronize(device)

    allocated_before_inf = backend.memory_allocated(device)
    backend.reset_peak_memory_stats(device)

    model.predict(text)
    backend.synchronize(device)

    allocated_after_inf = backend.memory_allocated(device)
    peak_inf = backend.max_memory_allocated(device)

    print("\n--- Inference (predict) ---")
    print(f"Base Allocated: {format_memory_to_mib(allocated_before_inf):.2f} MiB")
    print(f"Post Allocated: {format_memory_to_mib(allocated_after_inf):.2f} MiB")
    print(f"Peak Allocated: {format_memory_to_mib(peak_inf):.2f} MiB")
    print(f"Peak Delta: {format_memory_to_mib(peak_inf - allocated_before_inf):.2f} MiB")

    model.fit(text, label)  
    backend.synchronize(device)

    allocated_before_train = backend.memory_allocated(device)
    backend.reset_peak_memory_stats(device)

    model.fit(text, label)
    backend.synchronize(device)

    allocated_after_train = backend.memory_allocated(device)
    peak_train = backend.max_memory_allocated(device)

    print("\n--- Training Step (fit) ---")
    print(f"Base Allocated (with Adam states): {format_memory_to_mib(allocated_before_train):.2f} MiB")
    print(f"Post Allocated: {format_memory_to_mib(allocated_after_train):.2f} MiB")
    print(f"Peak Allocated: {format_memory_to_mib(peak_train):.2f} MiB")
    print(f"Peak Delta: {format_memory_to_mib(peak_train - allocated_before_train):.2f} MiB")


if __name__ == "__main__":
    run_vram_measurement()

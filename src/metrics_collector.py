import psutil
import torch
import time
from typing import Dict, Any

class MetricsCollector:
    """
    Collects system latency, CPU, and GPU memory metrics.
    """
    def __init__(self):
        self.start_time = 0
        self.end_time = 0

    def start_timer(self):
        self.start_time = time.perf_counter()

    def end_timer(self) -> float:
        self.end_time = time.perf_counter()
        return (self.end_time - self.start_time) * 1000  # Return ms

    def get_cpu_memory_mb(self) -> float:
        """Returns current CPU memory usage of the process in MB."""
        process = psutil.Process()
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024)

    def get_gpu_memory_mb(self) -> float:
        """Returns allocated GPU memory in MB. Returns 0.0 if CUDA is unavailable."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0

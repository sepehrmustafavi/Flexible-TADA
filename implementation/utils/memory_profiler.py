import logging
import time
import torch
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class HardwareProfiler:
    """
    A utility class for measuring Peak VRAM usage and Inference Latency.
    Crucial for extracting "Green AI" metrics for research papers (e.g., Table 4).
    """

    @staticmethod
    def reset_memory_stats():
        """Resets the PyTorch CUDA memory allocator statistics."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            logger.debug("CUDA memory stats reset.")
        else:
            logger.warning("CUDA is not available. Memory profiling will not work.")

    @staticmethod
    def get_peak_vram_mb() -> float:
        """
        Returns the peak VRAM used since the last reset, in Megabytes.
        """
        if torch.cuda.is_available():
            peak_bytes = torch.cuda.max_memory_allocated()
            peak_mb = peak_bytes / (1024 * 1024)
            return round(peak_mb, 2)
        return 0.0

    @staticmethod
    @contextmanager
    def track_vram_usage(operation_name: str = "Operation"):
        """
        A context manager to easily track VRAM usage of a specific block of code.
        
        Usage:
            with HardwareProfiler.track_vram_usage("Model Forward Pass"):
                outputs = model(inputs)
        """
        HardwareProfiler.reset_memory_stats()
        try:
            yield
        finally:
            peak_mb = HardwareProfiler.get_peak_vram_mb()
            logger.info(f"[VRAM Profiler] {operation_name} Peak VRAM: {peak_mb} MB")

    @staticmethod
    def measure_inference_latency(model, dummy_dataloader, device, num_batches=10):
        """
        Measures the exact inference latency per sample.
        Includes GPU warm-up and synchronization for scientific accuracy.

        Args:
            model: The PyTorch model.
            dummy_dataloader: A dataloader to fetch dummy batches.
            device: Target device (e.g., 'cuda:0').
            num_batches (int): Number of batches to run for the measurement.

        Returns:
            float: Average latency per sample in milliseconds.
        """
        model.eval()
        model.to(device)

        # 1. GPU Warm-up (Critical for accurate timing)
        # CUDA graphs and memory allocators need a few runs to stabilize.
        logger.info("Warming up GPU for latency measurement...")
        warmup_batches = min(3, len(dummy_dataloader))
        with torch.no_grad():
            for i, batch in enumerate(dummy_dataloader):
                if i >= warmup_batches:
                    break
                batch = {k: v.to(device) for k, v in batch.items() if k not in ["labels"]}
                _ = model(**batch)

        # 2. Actual Measurement using precise CUDA Events
        logger.info(f"Measuring latency over {num_batches} batches...")
        total_time_ms = 0.0
        total_samples = 0

        torch.cuda.synchronize() # Wait for warm-up to finish completely
        
        with torch.no_grad():
            for i, batch in enumerate(dummy_dataloader):
                if i >= num_batches:
                    break
                    
                batch = {k: v.to(device) for k, v in batch.items() if k not in ["labels"]}
                batch_size = batch["input_ids"].size(0)
                
                # Initialize precise GPU timers
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                _ = model(**batch)
                end_event.record()
                
                torch.cuda.synchronize() # Force CPU to wait for GPU to finish
                
                # Calculate elapsed time in ms directly from GPU hardware
                batch_time_ms = start_event.elapsed_time(end_event)
                total_time_ms += batch_time_ms
                total_samples += batch_size

        if total_samples == 0:
            return 0.0

        latency_per_sample = total_time_ms / total_samples
        logger.info(f"[Latency Profiler] Average Inference Latency: {latency_per_sample:.3f} ms/sample")
        
        return latency_per_sample
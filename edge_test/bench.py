"""
Benchmarking core: latency, throughput and memory for one model under one
resource constraint.

Everything here is a real measurement on the host. The device-specific scaling
lives in run_edge_test.py; keeping them apart is the point.
"""

import gc
import os
import platform
import time

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# host description
# --------------------------------------------------------------------------- #
def host_info():
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "logical_cores": os.cpu_count(),
        "torch_threads_default": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
    try:
        import psutil
        info["ram_total_mb"] = round(psutil.virtual_memory().total / 1e6)
    except ImportError:
        info["ram_total_mb"] = None
    return info


def process_rss_mb():
    """Resident set size in MB, or None without psutil."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except ImportError:
        return None


# --------------------------------------------------------------------------- #
# thread pinning
# --------------------------------------------------------------------------- #
class ThreadLimit:
    """Pin torch to n threads for the duration of a block, then restore.

    This is a genuine constraint, not a simulation: torch really does use only
    n threads inside the block. It is how the harness reproduces the core count
    of a small device.
    """

    def __init__(self, n):
        self.n = n
        self.prev = None
        self.prev_interop = None

    def __enter__(self):
        self.prev = torch.get_num_threads()
        if self.n:
            torch.set_num_threads(int(self.n))
            try:
                self.prev_interop = torch.get_num_interop_threads()
            except Exception:
                self.prev_interop = None
        return self

    def __exit__(self, *exc):
        torch.set_num_threads(self.prev)
        # interop threads cannot be changed after the pool starts; nothing to restore


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure(model, device, img_size=224, batch_size=1, runs=30, warmup=8,
            threads=None, capture_memory=True):
    """Median latency and throughput for one (model, constraint) pair.

    Returns a dict. Latency is per BATCH; ms_per_image divides by batch_size.
    Median rather than mean, because a single OS hiccup should not define the
    result; p90 is reported alongside so you can see the spread.
    """
    model = model.to(device).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    rss_before = process_rss_mb() if capture_memory else None

    with ThreadLimit(threads if device.type == "cpu" else None):
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

    times = np.array(times)
    rss_after = process_rss_mb() if capture_memory else None

    out = {
        "batch_size": batch_size,
        "img_size": img_size,
        "threads": threads or torch.get_num_threads(),
        "runs": runs,
        "ms_per_batch_median": float(np.median(times)),
        "ms_per_batch_p90": float(np.percentile(times, 90)),
        "ms_per_batch_min": float(times.min()),
        "ms_per_image": float(np.median(times)) / batch_size,
        "images_per_second": batch_size * 1000.0 / float(np.median(times)),
    }
    if device.type == "cuda":
        out["cuda_peak_mb"] = torch.cuda.max_memory_allocated() / 1e6
    if rss_before is not None and rss_after is not None:
        out["rss_delta_mb"] = round(rss_after - rss_before, 1)
        out["rss_after_mb"] = round(rss_after, 1)
    return out


def model_footprint(model, img_size=224):
    """Static cost of the model itself, independent of how fast it runs."""
    params = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers())
    param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    buffer_mb = sum(b.numel() * b.element_size() for b in model.buffers()) / 1e6

    # rough activation cost for one 224x224 image, measured by hooking every
    # module output once - this is what actually decides whether a small board
    # runs out of RAM, and it is usually bigger than the weights
    sizes = []

    def hook(_m, _i, o):
        if torch.is_tensor(o):
            sizes.append(o.numel() * o.element_size())

    handles = [m.register_forward_hook(hook) for m in model.modules()]
    try:
        with torch.no_grad():
            dev = next(model.parameters()).device
            model(torch.zeros(1, 3, img_size, img_size, device=dev))
    finally:
        for h in handles:
            h.remove()

    return {
        "params": params,
        "params_m": round(params / 1e6, 3),
        "buffers": buffers,
        "weights_mb_fp32": round(param_mb + buffer_mb, 2),
        "weights_mb_int8_est": round((param_mb + buffer_mb) / 4.0, 2),
        "activations_mb_fp32": round(sum(sizes) / 1e6, 2),
        "peak_working_set_mb_est": round(param_mb + buffer_mb + sum(sizes) / 1e6, 2),
    }


def try_dynamic_int8(model):
    """CPU dynamic quantisation of the Linear layers, or None if unsupported.

    Only the Linear layers - dynamic quantisation does not touch Conv2d, so the
    saving here is modest for these two convolutional models. It is included
    because it is the one quantisation that needs no calibration data, so it is
    the honest 'free' option; real edge INT8 would need static quantisation with
    a calibration set and op replacements for SiLU/SE.
    """
    try:
        import copy
        m = copy.deepcopy(model).cpu().eval()
        return torch.ao.quantization.quantize_dynamic(
            m, {torch.nn.Linear}, dtype=torch.qint8)
    except Exception as e:
        print("    dynamic int8 unavailable: %s" % e)
        return None

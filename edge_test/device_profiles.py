"""
Ten edge-device profiles for the simulated deployment test.

HOW THE SIMULATION WORKS - read this before quoting any number.

You cannot turn a desktop CPU into a Raspberry Pi. What this harness does instead
is two things, and it keeps them separate in the output:

1. MEASURED. Each device profile pins the run to a specific number of CPU threads
   and a specific input resolution, and the model is actually executed under that
   constraint on your machine. Thread count and resolution are real limits, so the
   scaling behaviour they produce is real - a 1-thread run genuinely costs what a
   1-thread run costs.

2. ESTIMATED. The remaining difference between your CPU core and the device's core
   (clock speed, IPC, memory bandwidth, SIMD width) is applied afterwards as a
   single multiplier, `cpu_factor`. That is a MODEL, not a measurement.

       estimated_ms = measured_ms_at_that_thread_count * cpu_factor

`cpu_factor` is "how many times slower is one core of this device than one core of
a modern x86 laptop", drawn from published single-core benchmark ratios. Treat the
estimates as an order-of-magnitude planning tool: they are good enough to tell you
"this model cannot run on a Pi 3" and not good enough to quote to three decimals.

If you have the real hardware, run `bench.py` on it and compare - and please fix
the factor here if it is off.

`ram_mb` is the usable RAM budget, deliberately below the sticker figure because
the OS takes its share. It is used only for the fits/does-not-fit verdict.
"""

# host reference: one core of a modern laptop/desktop x86 CPU = 1.0
DEVICES = [
    dict(
        key="pi_zero2w",
        name="Raspberry Pi Zero 2 W",
        soc="Broadcom BCM2710A1, 4x Cortex-A53 @ 1.0 GHz",
        threads=4, ram_mb=380, cpu_factor=14.0, accelerator=None,
        img_size=160,
        note="512 MB total. The teacher will not fit; this is the floor of the range.",
    ),
    dict(
        key="pi3b_plus",
        name="Raspberry Pi 3B+",
        soc="Broadcom BCM2837B0, 4x Cortex-A53 @ 1.4 GHz",
        threads=4, ram_mb=700, cpu_factor=10.0, accelerator=None,
        img_size=224,
        note="1 GB, no NEON-optimised BLAS by default. Still very common in the field.",
    ),
    dict(
        key="pi4",
        name="Raspberry Pi 4 (4 GB)",
        soc="Broadcom BCM2711, 4x Cortex-A72 @ 1.5 GHz",
        threads=4, ram_mb=3200, cpu_factor=6.5, accelerator=None,
        img_size=224,
        note="The realistic baseline for a field deployment today.",
    ),
    dict(
        key="pi5",
        name="Raspberry Pi 5 (8 GB)",
        soc="Broadcom BCM2712, 4x Cortex-A76 @ 2.4 GHz",
        threads=4, ram_mb=7000, cpu_factor=3.0, accelerator=None,
        img_size=224,
        note="Roughly 2-3x the Pi 4 per core.",
    ),
    dict(
        key="jetson_nano",
        name="NVIDIA Jetson Nano (4 GB)",
        soc="4x Cortex-A57 @ 1.43 GHz + 128-core Maxwell",
        threads=4, ram_mb=3000, cpu_factor=8.0, accelerator="gpu",
        accel_speedup=6.0, img_size=224,
        note="CPU is weak; the 472 GFLOPS Maxwell GPU is the point. accel_speedup is "
             "a conservative FP16 TensorRT estimate over its own CPU.",
    ),
    dict(
        key="jetson_orin_nano",
        name="NVIDIA Jetson Orin Nano (8 GB)",
        soc="6x Cortex-A78AE @ 1.5 GHz + 1024-core Ampere",
        threads=6, ram_mb=6500, cpu_factor=3.2, accelerator="gpu",
        accel_speedup=14.0, img_size=224,
        note="40 TOPS class. Comfortably real-time for both models.",
    ),
    dict(
        key="coral_dev",
        name="Google Coral Dev Board",
        soc="4x Cortex-A53 @ 1.5 GHz + Edge TPU",
        threads=4, ram_mb=800, cpu_factor=9.5, accelerator="tpu",
        accel_speedup=20.0, img_size=224,
        note="Edge TPU needs full INT8 quantisation and INT8-compatible ops. The "
             "student's SE blocks and SiLU would need replacing first - the "
             "accelerated figure here is what you would get AFTER that port, not "
             "what today's checkpoint does.",
    ),
    dict(
        key="phone_midrange",
        name="Mid-range Android phone",
        soc="Snapdragon 695, 2x A78 @ 2.2 GHz + 6x A55",
        threads=4, ram_mb=5000, cpu_factor=4.0, accelerator=None,
        img_size=224,
        note="Big cores only; thermal throttling ignored, so treat as optimistic.",
    ),
    dict(
        key="nuc_i5",
        name="Intel NUC (Core i5-1135G7)",
        soc="4 cores / 8 threads @ 2.4-4.2 GHz",
        threads=8, ram_mb=14000, cpu_factor=1.15, accelerator=None,
        img_size=224,
        note="A small x86 box - the 'edge server' end of the range.",
    ),
    dict(
        key="host",
        name="This machine (measured, unscaled)",
        soc="whatever you are running on",
        threads=None, ram_mb=None, cpu_factor=1.0, accelerator=None,
        img_size=224,
        note="Ground truth. cpu_factor 1.0 and all threads, so measured == estimated.",
    ),
]

# throughput bands used for the verdict column
FPS_BANDS = [
    (30.0, "real-time video"),
    (10.0, "interactive"),
    (2.0, "usable"),
    (0.5, "slow but works"),
    (0.0, "impractical"),
]


def fps_verdict(fps):
    for threshold, label in FPS_BANDS:
        if fps >= threshold:
            return label
    return "impractical"


def by_key(key):
    for d in DEVICES:
        if d["key"] == key:
            return d
    raise KeyError(key)

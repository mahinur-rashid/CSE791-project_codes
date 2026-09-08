# Edge deployment test - 10 simulated devices

Answers the question distillation is supposed to answer: **the student is 9x smaller, so where
can you actually run it that you could not run the teacher?**

```bash
pip install torch torchvision numpy matplotlib
pip install psutil            # optional, adds memory measurements
python run_edge_test.py
```

Takes a few minutes. Everything lands in `results/`.

## Read this before quoting a number

You cannot turn a desktop CPU into a Raspberry Pi, and this harness does not pretend to. It
separates what it measures from what it models, and keeps them in different columns:

| | what it is |
|---|---|
| `measured_ms_per_image` | **A real run on your machine**, pinned to that device's thread count and input resolution. Thread pinning is a genuine constraint (`torch.set_num_threads`), so the scaling it produces is real. |
| `est_ms_per_image` | `measured x cpu_factor`. **A model, not a measurement.** `cpu_factor` is how many times slower one core of that device is than one modern x86 core, taken from published single-core benchmarks. |

So the estimates are good enough to tell you *"the teacher cannot run on a Pi 3"* and not good
enough to quote to three decimals. If you have the hardware, run it there and correct the factor
in `device_profiles.py` - it is a plain dict, made to be edited.

## The ten devices

| key | device | threads | RAM budget | cpu_factor | accelerator |
|---|---|---|---|---|---|
| `pi_zero2w` | Raspberry Pi Zero 2 W | 4 | 380 MB | 14.0 | - |
| `pi3b_plus` | Raspberry Pi 3B+ | 4 | 700 MB | 10.0 | - |
| `pi4` | Raspberry Pi 4 (4 GB) | 4 | 3.2 GB | 6.5 | - |
| `pi5` | Raspberry Pi 5 (8 GB) | 4 | 7 GB | 3.0 | - |
| `jetson_nano` | NVIDIA Jetson Nano | 4 | 3 GB | 8.0 | GPU, ~6x |
| `jetson_orin_nano` | NVIDIA Jetson Orin Nano | 6 | 6.5 GB | 3.2 | GPU, ~14x |
| `coral_dev` | Google Coral Dev Board | 4 | 800 MB | 9.5 | Edge TPU, ~20x |
| `phone_midrange` | Snapdragon 695 phone | 4 | 5 GB | 4.0 | - |
| `nuc_i5` | Intel NUC i5-1135G7 | 8 | 14 GB | 1.15 | - |
| `host` | your machine, unscaled | all | - | 1.0 | - |

`host` is the control: factor 1.0, no thread limit, so measured and estimated are identical.
Every other row is that same measurement scaled.

## What comes out

```
results/
  edge_results.csv            one row per (device, model, batch size)
  edge_results.json           same, plus host info and model footprints
  report.md                   readable tables + a deployability verdict per model
  latency_by_device.png       estimated ms/image, log scale
  throughput_by_device.png    estimated fps, with real-time/interactive/usable bands
  model_cost.png              params, weight size, and host throughput side by side
```

The report ends with the line that matters for a write-up: which devices each model is usable on
(>=2 fps **and** fits in RAM) and which it is not.

## Memory, which is often the real blocker

Latency gets the attention, but on a 512 MB board the model simply will not load. `bench.py`
hooks every module once to measure the actual activation footprint at 224px, adds the weights,
and compares that working set against each device's RAM budget. The `fits_in_ram` column says
`NO` where it does not - the teacher fails this on the smallest boards regardless of how long
you are willing to wait.

## Options

```bash
python run_edge_test.py --quick                        # smoke test, 5 runs
python run_edge_test.py --devices pi4,jetson_nano,host # a subset
python run_edge_test.py --batch 1,4,8                  # batching helps throughput, not latency
python run_edge_test.py --runs 60                      # steadier medians
python run_edge_test.py --quantize                     # + dynamic-INT8 student on CPU
python run_edge_test.py --ckpt-dir <path>
```

Keep `--device cpu` (the default). Measuring on CUDA makes thread pinning meaningless and the
whole device simulation with it; the script warns you if you try.

## What the numbers are not

- **Thermal throttling is ignored.** A passively cooled Pi or a phone under sustained load will
  be slower than these figures, sometimes considerably.
- **The accelerator columns assume a port that has not been done.** The Jetson figures assume
  TensorRT FP16; the Coral figure assumes full INT8 with the SE blocks and SiLU activations
  replaced by Edge-TPU-supported ops. Today's checkpoints are plain fp32 PyTorch.
- **`--quantize` only touches Linear layers.** Dynamic quantisation does not quantise Conv2d, so
  on two convolutional models the saving is small. It is included because it is the only
  quantisation that needs no calibration data. Real edge INT8 means static quantisation with a
  calibration set, and it belongs in its own experiment.
- **Accuracy does not change across this table.** Teacher 0.9884, student 0.9846 on the kd07 run.
  The table is about what each of those accuracies costs to serve.

## Files

| file | what it is |
|---|---|
| `run_edge_test.py` | the driver - loops devices, writes CSV/JSON/report/charts |
| `bench.py` | measurement core: thread pinning, latency, memory footprint, INT8 helper |
| `device_profiles.py` | the ten profiles and their factors, with sourcing notes - edit freely |
| `kd_arch.py` | the two architectures + checkpoint loading (shared copy) |

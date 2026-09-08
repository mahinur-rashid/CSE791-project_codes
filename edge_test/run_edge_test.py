"""
Simulated edge deployment test: teacher vs student across 10 devices.

    pip install torch torchvision numpy matplotlib
    python run_edge_test.py

Optional:
    pip install psutil          # memory numbers
    python run_edge_test.py --devices pi4,jetson_nano,host
    python run_edge_test.py --runs 60 --batch 1,4
    python run_edge_test.py --quantize        # + CPU dynamic-INT8 student
    python run_edge_test.py --quick           # fewer runs, for a smoke test

Writes results/edge_results.csv, results/edge_results.json, results/report.md and
three charts under results/.

WHAT IS REAL AND WHAT IS MODELLED
---------------------------------
Each device profile pins the run to that device's core count and input resolution
and then actually executes the model under that limit on your machine - that part
is measured. The remaining per-core difference (clock, IPC, memory bandwidth) is
applied afterwards as one published-benchmark multiplier per device. So:

    measured_ms   real, on your CPU, at the device's thread count
    estimated_ms  measured_ms x cpu_factor        <- a model, not a measurement

Both are in the output, side by side, and the CSV keeps them in separate columns
so nothing gets quoted as measured when it is not. See device_profiles.py for the
factors and where they come from.
"""

import argparse
import csv
import json
import os
import time

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench import (host_info, measure, model_footprint, process_rss_mb,
                   try_dynamic_int8)
from device_profiles import DEVICES, fps_verdict
from kd_arch import count_params, load_student, load_teacher, pick_device


def human(n):
    return "%.1f" % n if n >= 10 else "%.2f" % n


# --------------------------------------------------------------------------- #
def run_device(profile, models, footprints, device, args):
    """Benchmark every model under one device profile."""
    rows = []
    threads = profile["threads"] or os.cpu_count()
    img_size = profile["img_size"]

    print("\n%s" % profile["name"])
    print("  %s" % profile["soc"])
    print("  threads %s | input %dpx | cpu_factor %.1fx%s"
          % (threads, img_size, profile["cpu_factor"],
             " | %s accel %.0fx" % (profile["accelerator"], profile.get("accel_speedup", 1))
             if profile.get("accelerator") else ""))

    for model_name, model in models.items():
        for bs in args.batch:
            m = measure(model, device, img_size=img_size, batch_size=bs,
                        runs=args.runs, warmup=args.warmup, threads=threads)

            factor = profile["cpu_factor"]
            est_ms_img = m["ms_per_image"] * factor
            est_fps = 1000.0 / est_ms_img

            row = {
                "device": profile["name"],
                "device_key": profile["key"],
                "soc": profile["soc"],
                "model": model_name,
                "batch_size": bs,
                "img_size": img_size,
                "threads": threads,
                "cpu_factor": factor,
                # --- measured on this host, at that thread count -------------
                "measured_ms_per_image": round(m["ms_per_image"], 3),
                "measured_ms_p90": round(m["ms_per_batch_p90"], 3),
                "measured_fps": round(m["images_per_second"], 2),
                # --- scaled estimate for the device --------------------------
                "est_ms_per_image": round(est_ms_img, 2),
                "est_fps": round(est_fps, 2),
                "est_verdict": fps_verdict(est_fps),
            }

            # accelerator path, where the device has one
            if profile.get("accelerator"):
                sp = profile.get("accel_speedup", 1.0)
                row["accelerator"] = profile["accelerator"]
                row["est_ms_per_image_accel"] = round(est_ms_img / sp, 2)
                row["est_fps_accel"] = round(est_fps * sp, 2)
                row["est_verdict_accel"] = fps_verdict(est_fps * sp)
            else:
                row["accelerator"] = ""
                row["est_ms_per_image_accel"] = ""
                row["est_fps_accel"] = ""
                row["est_verdict_accel"] = ""

            # memory verdict
            fp = footprints[model_name]
            need = fp["peak_working_set_mb_est"]
            row["working_set_mb"] = need
            if profile["ram_mb"]:
                row["ram_budget_mb"] = profile["ram_mb"]
                row["fits_in_ram"] = "yes" if need < profile["ram_mb"] else "NO"
            else:
                row["ram_budget_mb"] = ""
                row["fits_in_ram"] = "n/a"

            rows.append(row)
            print("    %-9s bs=%d  measured %8s ms/img  ->  device est %8s ms  %6s fps  %s%s"
                  % (model_name, bs, human(m["ms_per_image"]), human(est_ms_img),
                     human(est_fps), row["est_verdict"],
                     "" if row["fits_in_ram"] != "NO" else "   [DOES NOT FIT IN RAM]"))
    return rows


# --------------------------------------------------------------------------- #
def write_charts(rows, out_dir, models_footprint):
    bs1 = [r for r in rows if r["batch_size"] == 1]
    devices, seen = [], set()
    for r in bs1:
        if r["device"] not in seen:
            seen.add(r["device"])
            devices.append(r["device"])
    model_names = sorted({r["model"] for r in bs1})

    def series(model, key):
        return [next((r[key] for r in bs1 if r["device"] == d and r["model"] == model), 0)
                for d in devices]

    import numpy as np
    y = np.arange(len(devices))
    h = 0.8 / max(1, len(model_names))

    # 1. estimated latency
    fig, ax = plt.subplots(figsize=(10, 0.52 * len(devices) + 2.2))
    for i, m in enumerate(model_names):
        ax.barh(y + i * h, series(m, "est_ms_per_image"), height=h, label=m)
    ax.set_yticks(y + h * (len(model_names) - 1) / 2)
    ax.set_yticklabels(devices, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("estimated ms per image (log scale) - lower is better")
    ax.set_title("Estimated single-image latency by device")
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "latency_by_device.png"), dpi=130)
    plt.close(fig)

    # 2. estimated throughput with the usability bands
    fig, ax = plt.subplots(figsize=(10, 0.52 * len(devices) + 2.2))
    for i, m in enumerate(model_names):
        ax.barh(y + i * h, series(m, "est_fps"), height=h, label=m)
    for thr, lab in ((30, "real-time"), (10, "interactive"), (2, "usable")):
        ax.axvline(thr, ls="--", lw=1, color="grey", alpha=0.7)
        ax.text(thr, len(devices) - 0.3, " " + lab, fontsize=8, color="grey", rotation=90,
                va="top")
    ax.set_yticks(y + h * (len(model_names) - 1) / 2)
    ax.set_yticklabels(devices, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("estimated images per second (log scale) - higher is better")
    ax.set_title("Estimated throughput by device")
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "throughput_by_device.png"), dpi=130)
    plt.close(fig)

    # 3. the distillation trade: size, memory, speed
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    names = list(models_footprint.keys())
    axes[0].bar(names, [models_footprint[n]["params_m"] for n in names])
    axes[0].set_title("parameters (M)")
    axes[1].bar(names, [models_footprint[n]["weights_mb_fp32"] for n in names])
    axes[1].set_title("weights on disk (MB, fp32)")
    speeds = [next((r["measured_fps"] for r in bs1
                    if r["model"] == n and r["device_key"] == "host"), 0) for n in names]
    axes[2].bar(names, speeds)
    axes[2].set_title("images/sec on this host")
    for a in axes:
        a.tick_params(axis="x", labelsize=8)
        a.grid(axis="y", alpha=0.3)
    fig.suptitle("What distillation bought")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "model_cost.png"), dpi=130)
    plt.close(fig)


def write_report(rows, out_dir, info, models_footprint):
    lines = []
    A = lines.append
    A("# Edge deployment test - teacher vs distilled student\n")
    A("Generated %s\n" % time.strftime("%Y-%m-%d %H:%M"))

    A("\n## How to read this\n")
    A("`measured_*` columns are real runs on the host below, pinned to each device's thread")
    A("count and input resolution. `est_*` columns multiply those by a per-device single-core")
    A("factor from published benchmarks - a model, not a measurement. Order-of-magnitude")
    A("planning numbers, not spec-sheet claims. See `device_profiles.py`.\n")

    A("\n## Host\n")
    for k, v in info.items():
        A("- **%s**: %s" % (k, v))

    A("\n## The two models\n")
    A("| model | params | weights (fp32) | activations @224 | working set |")
    A("|---|---|---|---|---|")
    for n, f in models_footprint.items():
        A("| %s | %.2f M | %.1f MB | %.1f MB | %.1f MB |"
          % (n, f["params_m"], f["weights_mb_fp32"],
             f["activations_mb_fp32"], f["peak_working_set_mb_est"]))

    bs1 = [r for r in rows if r["batch_size"] == 1]
    A("\n## Single-image results (batch 1)\n")
    A("| device | model | measured ms | est. ms | est. fps | verdict | accel fps | RAM |")
    A("|---|---|---|---|---|---|---|---|")
    for r in bs1:
        A("| %s | %s | %.1f | %.1f | %.2f | %s | %s | %s |"
          % (r["device"], r["model"], r["measured_ms_per_image"], r["est_ms_per_image"],
             r["est_fps"], r["est_verdict"],
             r["est_fps_accel"] if r["est_fps_accel"] != "" else "-",
             r["fits_in_ram"]))

    ordered_devices = list(dict.fromkeys(r["device"] for r in bs1))
    A("\n## Speed-up from distillation\n")
    A("| device | teacher est. fps | student est. fps | student is |")
    A("|---|---|---|---|")
    for d in ordered_devices:
        t = next((r for r in bs1 if r["device"] == d and r["model"] == "teacher"), None)
        s = next((r for r in bs1 if r["device"] == d and r["model"] == "student"), None)
        if t and s and t["est_fps"] > 0:
            A("| %s | %.2f | %.2f | %.1fx faster |"
              % (d, t["est_fps"], s["est_fps"], s["est_fps"] / t["est_fps"]))

    A("\n## Where each model is deployable\n")
    for model in sorted({r["model"] for r in bs1}):
        ok = [r["device"] for r in bs1
              if r["model"] == model and r["est_fps"] >= 2.0 and r["fits_in_ram"] != "NO"]
        no = [r["device"] for r in bs1
              if r["model"] == model and (r["est_fps"] < 2.0 or r["fits_in_ram"] == "NO")]
        A("\n**%s**" % model)
        A("\n- usable (>=2 fps and fits in RAM): %s" % (", ".join(ok) if ok else "none"))
        A("- not usable: %s" % (", ".join(no) if no else "none"))

    A("\n## Caveats\n")
    A("- Thermal throttling is ignored. Sustained load on a passively cooled Pi or phone")
    A("  will be slower than these figures, sometimes by a lot.")
    A("- The accelerator columns assume a working TensorRT/Edge-TPU port. The current")
    A("  checkpoints are plain PyTorch fp32; the Coral figure in particular needs INT8")
    A("  quantisation and op replacements that this repo has not done.")
    A("- Estimates scale CPU compute only. Memory bandwidth and I/O limits on the smallest")
    A("  boards will make the real gap worse, not better.")
    A("- Accuracy is unchanged by any of this: teacher 0.9884, student 0.9846 on the kd07 run.")
    A("  The point of the table is what each accuracy costs to serve.\n")

    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", default="all",
                    help="comma-separated device keys, or 'all'")
    ap.add_argument("--batch", default="1",
                    help="comma-separated batch sizes, e.g. 1,4")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--device", default="cpu",
                    help="torch device for the measurements. cpu is the right choice: "
                         "edge devices are CPU-bound and thread pinning only works there")
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--quantize", action="store_true",
                    help="also benchmark a dynamic-INT8 student on CPU")
    ap.add_argument("--quick", action="store_true", help="fewer runs, for a smoke test")
    args = ap.parse_args()

    if args.quick:
        args.runs, args.warmup = 5, 2
    args.batch = [int(b) for b in args.batch.split(",")]

    device = pick_device(args.device)
    if device.type == "cuda":
        print("NOTE: measuring on CUDA. Thread pinning does nothing there, so the device\n"
              "      simulation is meaningless. Use --device cpu for the edge table.")

    profiles = DEVICES if args.devices == "all" else [
        d for d in DEVICES if d["key"] in args.devices.split(",")]
    if not profiles:
        raise SystemExit("no matching device keys. available: %s"
                         % ", ".join(d["key"] for d in DEVICES))

    info = host_info()
    print("=" * 76)
    print("HOST")
    for k, v in info.items():
        print("  %-22s %s" % (k, v))
    print("=" * 76)

    print("\nloading checkpoints...")
    teacher = load_teacher(args.ckpt_dir, device=device)
    student = load_student(args.ckpt_dir, device=device)
    models = {"teacher": teacher, "student": student}

    if args.quantize and device.type == "cpu":
        q = try_dynamic_int8(student)
        if q is not None:
            models["student-int8"] = q

    models_footprint = {n: model_footprint(m) for n, m in models.items()}
    print("\nmodel footprints")
    for n, f in models_footprint.items():
        print("  %-14s %6.2f M params | weights %6.1f MB | activations %6.1f MB | working set %6.1f MB"
              % (n, f["params_m"], f["weights_mb_fp32"],
                 f["activations_mb_fp32"], f["peak_working_set_mb_est"]))
    print("\ncompression: %.1fx fewer parameters in the student"
          % (count_params(teacher) / count_params(student)))

    rows = []
    t0 = time.time()
    for p in profiles:
        rows.extend(run_device(p, models, models_footprint, device, args))

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "edge_results.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(out_dir, "edge_results.json"), "w", encoding="utf-8") as f:
        json.dump({"host": info, "footprints": models_footprint,
                   "settings": {"runs": args.runs, "warmup": args.warmup,
                                "batch_sizes": args.batch, "torch_device": str(device)},
                   "rows": rows}, f, indent=2)

    write_charts(rows, out_dir, models_footprint)
    write_report(rows, out_dir, info, models_footprint)

    print("\n" + "=" * 76)
    print("done in %.1f s" % (time.time() - t0))
    print("  %s" % os.path.join(out_dir, "edge_results.csv"))
    print("  %s" % os.path.join(out_dir, "report.md"))
    print("  %s" % os.path.join(out_dir, "latency_by_device.png"))
    print("  %s" % os.path.join(out_dir, "throughput_by_device.png"))
    print("  %s" % os.path.join(out_dir, "model_cost.png"))
    print("=" * 76)


if __name__ == "__main__":
    main()

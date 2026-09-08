"""
Grad-CAM explanations for the kd07 pair, teacher vs student, side by side.

    python run_xai.py                 # 500 images, one PNG each
    python run_xai.py --n 50          # quick look
    python run_xai.py --branches      # add the teacher's two branch CAMs
    python run_xai.py --attention     # add the student's learned attention map

Every image produces:

    outputs/panels/<i>_<true>_T-<pred>_S-<pred>.png     original | teacher | student
    outputs/teacher_cam/<same name>.png                 teacher overlay alone
    outputs/student_cam/<same name>.png                 student overlay alone

plus outputs/summary.csv and outputs/summary.json with, per image, both models'
predictions and two agreement measures between the CAMs themselves:

    cam_cosine  - overall similarity of the two heatmaps
    cam_iou     - overlap of the top-20% hottest regions

Those two are the interesting numbers for a distillation write-up: accuracy says
the student reaches the same answer, cam_iou says whether it got there by looking
at the same part of the leaf.
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")                     # no display needed
import matplotlib.pyplot as plt

from kd_arch import (CLASS_NAMES, IMG_SIZE, denormalize, eval_transform,
                     load_pair, pick_device)
from gradcam import GradCAM, TeacherCAM, cam_cosine, cam_iou, overlay

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# --------------------------------------------------------------------------- #
# dataset + the same split the checkpoint was trained under
# --------------------------------------------------------------------------- #
def find_dataset_dir(explicit=None):
    here = os.path.dirname(os.path.abspath(__file__))
    for c in [explicit,
              os.environ.get("RICE_DATASET_DIR"),
              os.path.join(here, "..", "Augmented Images"),
              r"E:\riceleaf\Augmented Images"]:
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    raise FileNotFoundError("dataset not found - pass --data-dir")


def scan_dataset(root):
    """ImageFolder-style scan -> (paths, labels, class_names), sorted like training."""
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    paths, labels = [], []
    for ci, c in enumerate(classes):
        d = os.path.join(root, c)
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(IMG_EXT):
                paths.append(os.path.join(d, f))
                labels.append(ci)
    return np.array(paths), np.array(labels), classes


def held_out_indices(labels, scheme="kd07", seed=42):
    """Indices of the images the kd07 checkpoint did NOT train on.

    scheme="kd07"  reproduces the 85/15 split that run actually used (the saved
                   weights come from a Kaggle run made before the 60/20/20 change),
                   so the returned indices are genuinely unseen by these weights.
    scheme="60-20-20" returns the test fifth of the newer split.
    scheme="all"   ignores splitting - useful only for eyeballing.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    idx_all = np.arange(len(labels))
    if scheme == "all":
        return idx_all
    if scheme == "kd07":
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        _, val_idx = next(sss.split(np.zeros(len(labels)), labels))
        return val_idx
    if scheme == "60-20-20":
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        _, test_idx = next(sss.split(np.zeros(len(labels)), labels))
        return test_idx
    raise ValueError("unknown scheme %r" % scheme)


def stratified_sample(indices, labels, n, seed=42):
    """Pick n indices spread evenly over the classes present."""
    rng = np.random.RandomState(seed)
    if n >= len(indices):
        return indices
    by_class = {}
    for i in indices:
        by_class.setdefault(int(labels[i]), []).append(i)
    classes = sorted(by_class)
    per = max(1, n // len(classes))
    picked = []
    for c in classes:
        pool = np.array(by_class[c])
        take = min(per, len(pool))
        picked.extend(rng.choice(pool, size=take, replace=False).tolist())
    # top up (or trim) to exactly n
    if len(picked) < n:
        rest = np.array([i for i in indices if i not in set(picked)])
        if len(rest):
            extra = rng.choice(rest, size=min(n - len(picked), len(rest)), replace=False)
            picked.extend(extra.tolist())
    picked = sorted(picked)[:n]
    return np.array(picked)


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def short(name, width=18):
    return name if len(name) <= width else name[:width - 1] + "\u2026"


def save_panel(path, img, t_over, s_over, meta, extra=None, dpi=110):
    """One row: original | teacher CAM | student CAM | (optional extras)."""
    cols = [("input", img, None), ("teacher Grad-CAM", t_over, meta["t_label"]),
            ("student Grad-CAM", s_over, meta["s_label"])]
    if extra:
        cols.extend(extra)

    fig, axes = plt.subplots(1, len(cols), figsize=(3.1 * len(cols), 3.7))
    axes = np.atleast_1d(axes)
    for ax, (title, arr, sub) in zip(axes, cols):
        ax.imshow(arr)
        ax.set_title(title, fontsize=10)
        if sub:
            ax.set_xlabel(sub, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    agree = "agree" if meta["agree"] else "DISAGREE"
    ok = "correct" if meta["t_correct"] and meta["s_correct"] else (
        "teacher only" if meta["t_correct"] else
        ("student only" if meta["s_correct"] else "both wrong"))
    fig.suptitle("true: %s     |     %s     |     %s     |     CAM IoU %.2f  cos %.2f"
                 % (meta["true_name"], agree, ok, meta["cam_iou"], meta["cam_cosine"]),
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_single(path, arr, dpi=110):
    fig = plt.figure(figsize=(3.0, 3.0))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(arr)
    ax.set_axis_off()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=500, help="how many images (default 500)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--split", default="kd07", choices=("kd07", "60-20-20", "all"),
                    help="which images to explain; kd07 = unseen by these weights")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--branches", action="store_true",
                    help="add one panel per teacher branch (effnet / convnext)")
    ap.add_argument("--attention", action="store_true",
                    help="add the student's learned spatial-attention map")
    ap.add_argument("--alpha", type=float, default=0.45, help="heatmap opacity")
    ap.add_argument("--no-singles", action="store_true",
                    help="only write the combined panels, not the separate overlays")
    ap.add_argument("--dpi", type=int, default=110,
                    help="figure resolution. 500 panels at 110 dpi is roughly 150-400 MB; "
                         "drop to 80 for a lighter run")
    args = ap.parse_args()

    device = pick_device(args.device)
    data_dir = find_dataset_dir(args.data_dir)
    paths, labels, classes = scan_dataset(data_dir)
    if classes != CLASS_NAMES:
        print("NOTE: folder classes differ from the trained CLASS_NAMES.")
        print("  on disk :", classes)
        print("  trained :", CLASS_NAMES)
        print("  predictions are indexed by the TRAINED order.")

    pool = held_out_indices(labels, scheme=args.split, seed=args.seed)
    chosen = stratified_sample(pool, labels, args.n, seed=args.seed)

    print("dataset   ->", data_dir)
    print("device    ->", device)
    print("split     -> %s (%d images available)" % (args.split, len(pool)))
    print("explaining %d images" % len(chosen))

    teacher, student = load_pair(args.ckpt_dir, device=device)

    out = os.path.abspath(args.out)
    dirs = {k: os.path.join(out, k) for k in ("panels", "teacher_cam", "student_cam")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    tf = eval_transform(IMG_SIZE)
    rows = []
    t0 = time.time()

    with TeacherCAM(teacher) as tcam, GradCAM(student, student.cam_layers()[0]) as scam:
        for n, idx in enumerate(chosen, 1):
            path, y = paths[idx], int(labels[idx])
            pil = Image.open(path).convert("RGB")
            x = tf(pil).unsqueeze(0).to(device)

            # each model explains ITS OWN prediction - that is the honest comparison
            t_out = tcam(x)
            t_probs = torch.softmax(t_out["logits"], dim=1)[0]
            t_pred = int(t_out["class_idx"][0])

            s_cam, s_logits, s_idx = scam(x)
            s_probs = torch.softmax(s_logits, dim=1)[0]
            s_pred = int(s_idx[0])

            img = denormalize(x[0])
            t_map = t_out["fused"][0, 0].detach().cpu().numpy()
            s_map = s_cam[0, 0].detach().cpu().numpy()
            t_over = overlay(img, t_map, args.alpha)
            s_over = overlay(img, s_map, args.alpha)

            meta = {
                "true_name": CLASS_NAMES[y] if y < len(CLASS_NAMES) else str(y),
                "t_label": "%s  %.1f%%" % (short(CLASS_NAMES[t_pred]), 100 * t_probs[t_pred]),
                "s_label": "%s  %.1f%%" % (short(CLASS_NAMES[s_pred]), 100 * s_probs[s_pred]),
                "agree": t_pred == s_pred,
                "t_correct": t_pred == y,
                "s_correct": s_pred == y,
                "cam_cosine": cam_cosine(t_map, s_map),
                "cam_iou": cam_iou(t_map, s_map, q=0.80),
            }

            extra = []
            if args.branches:
                gate = t_out["gate"][0]
                for bi, bname in enumerate(t_out["branch_names"]):
                    bmap = t_out["branches"][bi][0, 0].detach().cpu().numpy()
                    extra.append(("teacher: " + bname,
                                  overlay(img, bmap, args.alpha),
                                  "gate %.2f" % gate[bi]))
            if args.attention:
                att = student.attention_map(x)
                att = torch.nn.functional.interpolate(
                    att, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
                att = att[0, 0].detach().cpu().numpy()
                att = (att - att.min()) / (att.max() - att.min() + 1e-8)
                extra.append(("student attention", overlay(img, att, args.alpha),
                              "learned pooling"))

            stem = "%04d_%s_T-%s_S-%s" % (
                n, CLASS_NAMES[y].replace(" ", ""),
                CLASS_NAMES[t_pred].replace(" ", ""), CLASS_NAMES[s_pred].replace(" ", ""))
            save_panel(os.path.join(dirs["panels"], stem + ".png"),
                       img, t_over, s_over, meta, extra, dpi=args.dpi)
            if not args.no_singles:
                save_single(os.path.join(dirs["teacher_cam"], stem + ".png"), t_over,
                            dpi=args.dpi)
                save_single(os.path.join(dirs["student_cam"], stem + ".png"), s_over,
                            dpi=args.dpi)

            rows.append({
                "n": n, "file": os.path.relpath(path, data_dir),
                "true": CLASS_NAMES[y],
                "teacher_pred": CLASS_NAMES[t_pred],
                "teacher_conf": round(float(t_probs[t_pred]), 4),
                "student_pred": CLASS_NAMES[s_pred],
                "student_conf": round(float(s_probs[s_pred]), 4),
                "teacher_correct": int(meta["t_correct"]),
                "student_correct": int(meta["s_correct"]),
                "agree": int(meta["agree"]),
                "cam_cosine": round(meta["cam_cosine"], 4),
                "cam_iou": round(meta["cam_iou"], 4),
                "gate_%s" % t_out["branch_names"][0]: round(float(t_out["gate"][0, 0]), 4),
                "gate_%s" % t_out["branch_names"][1]: round(float(t_out["gate"][0, 1]), 4),
                "panel": os.path.join("panels", stem + ".png"),
            })

            if n % 25 == 0 or n == len(chosen):
                print("  %4d/%d  (%.1fs)" % (n, len(chosen), time.time() - t0))

    # ---- summary ---------------------------------------------------------- #
    t_acc = float(np.mean([r["teacher_correct"] for r in rows]))
    s_acc = float(np.mean([r["student_correct"] for r in rows]))
    agree = float(np.mean([r["agree"] for r in rows]))
    m_cos = float(np.mean([r["cam_cosine"] for r in rows]))
    m_iou = float(np.mean([r["cam_iou"] for r in rows]))

    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "images": len(rows), "split": args.split, "dataset": data_dir,
        "teacher_accuracy": round(t_acc, 4), "student_accuracy": round(s_acc, 4),
        "prediction_agreement": round(agree, 4),
        "mean_cam_cosine": round(m_cos, 4), "mean_cam_iou": round(m_iou, 4),
        "rows": rows,
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 62)
    print("images explained        %d" % len(rows))
    print("teacher accuracy        %.4f" % t_acc)
    print("student accuracy        %.4f" % s_acc)
    print("they predict the same   %.4f of the time" % agree)
    print("mean CAM cosine         %.4f   (heatmap similarity)" % m_cos)
    print("mean CAM IoU@top20%%     %.4f   (do they look at the same spot)" % m_iou)
    print("=" * 62)
    print("panels  ->", dirs["panels"])
    print("summary ->", os.path.join(out, "summary.csv"))


if __name__ == "__main__":
    main()

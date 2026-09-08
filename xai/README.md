# XAI - Grad-CAM, teacher vs student

Explains the **kd07** pair on 500 images and writes one figure per image.

| | model | params | val accuracy (kd07 run) |
|---|---|---|---|
| teacher | `DualBranchFusionTeacher` (EfficientNet-B3 + ConvNeXt-Tiny, learned gate) | 40.1 M | 0.9884 |
| student | `RiceNetStudent` (MobileNetV3-Large stem) | 4.5 M | 0.9846 |

## Run it

```bash
pip install torch torchvision pillow numpy matplotlib scikit-learn
python run_xai.py
```

That is all it needs - the checkpoints and dataset are found automatically
(`../results/checkpoints`, `../Augmented Images`). Useful flags:

```bash
python run_xai.py --n 50            # quick look before committing to 500
python run_xai.py --branches        # + one panel per teacher branch, with its gate weight
python run_xai.py --attention       # + the student's own learned attention map
python run_xai.py --split all       # explain any image, not just held-out ones
python run_xai.py --device cpu
python run_xai.py --ckpt-dir <path> --data-dir <path>
python run_xai.py --dpi 80 --no-singles   # lighter on disk
```

**Disk:** 500 images at the default 110 dpi writes three PNGs each and lands somewhere around
150-400 MB depending on how busy the photos are. `--dpi 80` roughly halves it and `--no-singles`
skips the two per-model overlays, keeping only the combined panels.

## What comes out

```
outputs/
  panels/       0001_LeafBlast_T-LeafBlast_S-LeafBlast.png     <- original | teacher | student
  teacher_cam/  0001_...png                                    <- teacher overlay on its own
  student_cam/  0001_...png                                    <- student overlay on its own
  summary.csv
  summary.json
```

Each panel is one row: the input, the teacher's Grad-CAM, the student's Grad-CAM, with each
model's own prediction and confidence underneath, and a title line giving the true class,
whether the two models agreed, who got it right, and the two CAM similarity numbers.

## Which images

By default, the 500 come from the **15% the kd07 weights never trained on**. Those weights come
from a Kaggle run made before the folder moved to 60/20/20, so `--split kd07` reproduces that
run's exact 85/15 stratified split (seed 42) and samples from its validation side. Explaining
training images instead would flatter the models, so this matters.

`--split 60-20-20` uses the test fifth of the newer split, and `--split all` ignores splitting.
The sample is stratified, so all eight classes are represented roughly equally.

## Reading the two CAM numbers

Grad-CAM answers "which pixels moved this logit". Comparing the teacher's map with the student's
answers a sharper question than accuracy does:

- **`cam_cosine`** - overall similarity of the two heatmaps.
- **`cam_iou`** - overlap of the top-20% hottest region of each. This is the strict one: it asks
  whether the two models are pointing at the *same lesion*, not just producing similar-looking
  blobs.

Accuracy already says the student reaches the teacher's answer 98% of the time. A high `cam_iou`
says it also got there for the same reason - which is the claim distillation actually makes.
A high agreement rate with a low `cam_iou` is the interesting failure: same answers, different
evidence, and worth a paragraph in the write-up.

`summary.csv` has one row per image, so you can sort by `cam_iou` to find the clearest examples
of both cases, and the panel filename for each is in the last column.

## How the teacher's CAM is built

The teacher has two backbones and a learned gate that weights them **per image**, so it has no
single feature map to hook. `TeacherCAM` taps both trunks, runs one forward and one backward, and
fuses the two branch CAMs using the gate weights the model itself produced for that image. Run
with `--branches` to see the two separately along with their gate values - watching the gate shift
between EfficientNet and ConvNeXt across classes is a result in itself.

The student is simpler: the hooked layer is `head_blocks`, the last spatial map before its
attention pooling. `--attention` additionally plots that pooling's own softmax map, which is
learned rather than derived from gradients.

## Files

| file | what it is |
|---|---|
| `run_xai.py` | the script - selection, loop, figures, summary |
| `gradcam.py` | `GradCAM`, `TeacherCAM`, overlay rendering, `cam_cosine`, `cam_iou` |
| `kd_arch.py` | the two architectures + checkpoint loading (copy of the trained definitions) |

`kd_arch.py` is duplicated in `app/` and `edge_test/` so each folder stands alone. It must stay in
sync with `kd_aug/kd_models.py` or the checkpoints will not load.

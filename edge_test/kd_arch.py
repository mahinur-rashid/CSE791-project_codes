"""
Model definitions + checkpoint loader for the kd07 pair.

Self-contained on purpose: this file is copied into xai/, app/ and edge_test/ so
each folder runs on its own without importing from kd_aug/. The architectures are
byte-for-byte the ones in kd_aug/kd_models.py - change them and the checkpoints
stop loading.

    teacher : DualBranchFusionTeacher(efficientnet_b3 + convnext_tiny), 40.1 M
    student : RiceNetStudent(mobilenet_v3_large stem),                   4.5 M
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

# --------------------------------------------------------------------------- #
# dataset / preprocessing constants - must match training
# --------------------------------------------------------------------------- #
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf scald",
    "Narrow Brown Leaf Spot",
    "Rice Hispa",
    "Sheath Blight",
]
NUM_CLASSES = len(CLASS_NAMES)

TEACHER_CKPT = "teacher_fusion_effb3_convnext.pth"
STUDENT_CKPT = "kd07_fusion_ricenet_student.pth"
BASELINE_CKPT = "kd07_fusion_ricenet_baseline.pth"


def eval_transform(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(t):
    """(3,H,W) normalised tensor -> (H,W,3) float array in [0,1], for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t.detach().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


# --------------------------------------------------------------------------- #
# building blocks (identical to kd_models.py)
# --------------------------------------------------------------------------- #
class SqueezeExcite(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        hidden = max(8, ch // r)
        self.fc1 = nn.Conv2d(ch, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, ch, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.silu(self.fc1(s))
        return x * torch.sigmoid(self.fc2(s))


class DSBlock(nn.Module):
    """Depthwise-separable block with SE and a residual when shapes allow."""

    def __init__(self, cin, cout, stride=1, expand=4):
        super().__init__()
        mid = cin * expand
        self.use_res = (stride == 1 and cin == cout)
        self.block = nn.Sequential(
            nn.Conv2d(cin, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.SiLU(inplace=True),
            nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False),
            nn.BatchNorm2d(mid), nn.SiLU(inplace=True),
            SqueezeExcite(mid),
            nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout),
        )

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_res else out


class SpatialAttentionPool(nn.Module):
    """Learned spatial attention + avg/max pooling -> one feature vector."""

    def __init__(self, cin, hidden=128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv2d(cin, hidden, 1, bias=False), nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True), nn.Conv2d(hidden, 1, 1),
        )

    def attention_map(self, x):
        """The softmax attention map itself, (B,1,H,W) - free interpretability."""
        b, c, h, w = x.shape
        a = self.score(x).flatten(2)
        return torch.softmax(a, dim=2).view(b, 1, h, w)

    def forward(self, x):
        a = self.attention_map(x)
        attn = (x * a).sum(dim=(2, 3))
        mx = F.adaptive_max_pool2d(x, 1).flatten(1)
        return torch.cat([attn, mx], dim=1)


def strip_head(model):
    """Replace the final Linear with Identity so the model outputs features."""
    for attr in ("heads", "head", "classifier", "fc"):
        if not hasattr(model, attr):
            continue
        mod = getattr(model, attr)
        if isinstance(mod, nn.Linear):
            dim = mod.in_features
            setattr(model, attr, nn.Identity())
            return model, dim
        if isinstance(mod, nn.Sequential):
            lin_idx = [i for i, m in enumerate(mod) if isinstance(m, nn.Linear)]
            if lin_idx:
                i = lin_idx[-1]
                dim = mod[i].in_features
                mod[i] = nn.Identity()
                return model, dim
        if hasattr(mod, "head") and isinstance(mod.head, nn.Linear):
            dim = mod.head.in_features
            mod.head = nn.Identity()
            return model, dim
    raise ValueError("Could not locate a classifier head on %s" % type(model).__name__)


def _backbone_features(name, pretrained=False):
    """Conv trunk that outputs a 4D (B,C,H,W) map, plus its width."""
    m = torchvision.models.get_model(name, weights="DEFAULT" if pretrained else None)
    if hasattr(m, "features"):
        trunk = m.features
    else:
        trunk = nn.Sequential(*list(m.children())[:-2])
    with torch.no_grad():
        ch = trunk(torch.zeros(1, 3, 64, 64)).shape[1]
    return trunk, ch


# --------------------------------------------------------------------------- #
# TEACHER
# --------------------------------------------------------------------------- #
class DualBranchFusionTeacher(nn.Module):
    """Two pretrained backbones, a learned gate over their pooled features, custom head."""

    def __init__(self, num_classes=NUM_CLASSES,
                 backbones=("efficientnet_b3", "convnext_tiny"),
                 pretrained=False, embed=512, dropout=0.3):
        super().__init__()
        branches, dims = [], []
        for name in backbones:
            m = torchvision.models.get_model(name, weights="DEFAULT" if pretrained else None)
            m, d = strip_head(m)
            branches.append(m)
            dims.append(d)
        self.branches = nn.ModuleList(branches)
        self.dims = dims
        self.backbone_names = tuple(backbones)

        self.head_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(d, embed), nn.LayerNorm(embed), nn.GELU()) for d in dims
        ])
        self.head_gate = nn.Sequential(
            nn.Linear(sum(dims), 128), nn.GELU(), nn.Linear(128, len(dims))
        )
        self.head_embed = nn.LayerNorm(embed)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(embed, embed // 2), nn.GELU(),
            nn.Dropout(dropout / 2), nn.Linear(embed // 2, num_classes),
        )
        self.embed_dim = embed
        self.model_name = "fusion(" + "+".join(backbones) + ")"

    def forward_features(self, x):
        feats = [b(x) for b in self.branches]
        gate = torch.softmax(self.head_gate(torch.cat(feats, dim=1)), dim=1)
        proj = torch.stack([p(f) for p, f in zip(self.head_proj, feats)], dim=1)
        fused = (proj * gate.unsqueeze(-1)).sum(dim=1)
        return self.head_embed(fused)

    def forward(self, x):
        return self.head(self.forward_features(x))

    @torch.no_grad()
    def gate_weights(self, x):
        """Diagnostic: how much each branch is trusted, per image. (B, n_branches)"""
        feats = [b(x) for b in self.branches]
        return torch.softmax(self.head_gate(torch.cat(feats, dim=1)), dim=1)

    def cam_layers(self):
        """The last spatial feature map of each branch - Grad-CAM targets."""
        return [b.features for b in self.branches]


# --------------------------------------------------------------------------- #
# STUDENT
# --------------------------------------------------------------------------- #
class RiceNetStudent(nn.Module):
    """Pretrained lightweight stem -> custom DS blocks -> attention pooling -> head."""

    def __init__(self, num_classes=NUM_CLASSES, backbone="mobilenet_v3_large",
                 pretrained=False, stage_ch=256, embed=256, dropout=0.2, n_blocks=2):
        super().__init__()
        self.encoder, ch = _backbone_features(backbone, pretrained)
        self.head_reduce = nn.Sequential(
            nn.Conv2d(ch, stage_ch, 1, bias=False), nn.BatchNorm2d(stage_ch),
            nn.SiLU(inplace=True),
        )
        self.head_blocks = nn.Sequential(
            *[DSBlock(stage_ch, stage_ch, stride=1, expand=3) for _ in range(n_blocks)]
        )
        self.head_pool = SpatialAttentionPool(stage_ch, hidden=64)
        self.head_embed = nn.Sequential(
            nn.Linear(2 * stage_ch, embed), nn.BatchNorm1d(embed), nn.SiLU(inplace=True)
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed, num_classes))
        self.embed_dim = embed
        self.model_name = "ricenet(" + backbone + ")"

    def spatial_features(self, x):
        """(B, stage_ch, H, W) - what the attention pooling and Grad-CAM see."""
        return self.head_blocks(self.head_reduce(self.encoder(x)))

    def forward_features(self, x):
        return self.head_embed(self.head_pool(self.spatial_features(x)))

    def forward(self, x):
        return self.head(self.forward_features(x))

    @torch.no_grad()
    def attention_map(self, x):
        """The student's own learned 'where to look' map, (B,1,H,W)."""
        return self.head_pool.attention_map(self.spatial_features(x))

    def cam_layers(self):
        return [self.head_blocks]


# --------------------------------------------------------------------------- #
# checkpoint loading
# --------------------------------------------------------------------------- #
def find_ckpt_dir(explicit=None):
    """First existing candidate: explicit arg, env var, then the usual places."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        explicit,
        os.environ.get("RICE_KD_CKPT_DIR"),
        os.path.join(here, "checkpoints"),
        os.path.join(here, "..", "results", "checkpoints"),
        r"E:\riceleaf\results\checkpoints",
        os.path.join(here, "..", "kd_aug", "checkpoints"),
    ]
    for c in candidates:
        if c and os.path.isdir(c) and os.path.exists(os.path.join(c, TEACHER_CKPT)):
            return os.path.abspath(c)
    tried = "\n  ".join(str(c) for c in candidates if c)
    raise FileNotFoundError(
        "could not find %s in any of:\n  %s\n"
        "Pass --ckpt-dir, or set RICE_KD_CKPT_DIR." % (TEACHER_CKPT, tried))


def _load_into(model, path, device):
    if not os.path.exists(path):
        raise FileNotFoundError("checkpoint not found: %s" % path)
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    return model


def load_teacher(ckpt_dir=None, device="cpu", num_classes=NUM_CLASSES):
    ckpt_dir = find_ckpt_dir(ckpt_dir)
    m = DualBranchFusionTeacher(num_classes, pretrained=False)
    _load_into(m, os.path.join(ckpt_dir, TEACHER_CKPT), device)
    return m.to(device).eval()


def load_student(ckpt_dir=None, device="cpu", num_classes=NUM_CLASSES, distilled=True):
    ckpt_dir = find_ckpt_dir(ckpt_dir)
    name = STUDENT_CKPT if distilled else BASELINE_CKPT
    m = RiceNetStudent(num_classes, pretrained=False)
    _load_into(m, os.path.join(ckpt_dir, name), device)
    return m.to(device).eval()


def load_pair(ckpt_dir=None, device="cpu", verbose=True):
    """(teacher, student) both in eval mode on `device`."""
    ckpt_dir = find_ckpt_dir(ckpt_dir)
    teacher = load_teacher(ckpt_dir, device)
    student = load_student(ckpt_dir, device)
    if verbose:
        print("checkpoints  ->", ckpt_dir)
        print("teacher      -> %s  %.1f M params" % (teacher.model_name, count_params(teacher) / 1e6))
        print("student      -> %s  %.2f M params" % (student.model_name, count_params(student) / 1e6))
    return teacher, student


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def pick_device(arg="auto"):
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)

"""
Grad-CAM for the kd07 teacher/student pair.

Standard Grad-CAM (Selvaraju et al.): hook a spatial feature map, backprop the
score of one class into it, weight each channel by the global average of its
gradient, sum, ReLU, upsample.

Two wrinkles this project needs:

* the teacher is a TWO-BRANCH network with a learned gate, so a single CAM does
  not exist. `TeacherCAM` computes one CAM per branch from a single backward pass
  and fuses them with the gate weights the model itself produced for that image.
* ConvNeXt/Swin-style trunks can emit channels-last maps, so layout is detected
  from the tensor shape rather than assumed.
"""

import numpy as np
import torch
import torch.nn.functional as F


def _to_bchw(t):
    """(B,C,H,W) or (B,H,W,C) -> (B,C,H,W). Heuristic: channels are the big axis."""
    if t.dim() == 4 and t.shape[1] < t.shape[-1]:
        return t.permute(0, 3, 1, 2).contiguous()
    return t


def _normalize(cam, eps=1e-8):
    """Per-image min-max to [0,1].

    A naive `/(hi - lo + eps)` quietly rescales the whole map when the range is
    itself near eps, turning a flat CAM into a dim one instead of an empty one.
    Dividing by 1.0 in that degenerate case keeps "this layer carried no signal"
    looking like no signal.
    """
    b = cam.shape[0]
    flat = cam.view(b, -1)
    lo = flat.min(dim=1)[0].view(b, 1, 1, 1)
    hi = flat.max(dim=1)[0].view(b, 1, 1, 1)
    rng = hi - lo
    denom = torch.where(rng > eps, rng, torch.ones_like(rng))
    return ((cam - lo) / denom).clamp(0.0, 1.0)


class _Tap:
    """Captures a module's output and the gradient flowing back through it."""

    def __init__(self, module):
        self.activations = None
        self.gradients = None
        self._handle = module.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        out = output[0] if isinstance(output, (list, tuple)) else output
        self.activations = out
        if out.requires_grad:
            out.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self.gradients = grad

    def cam(self, out_size):
        """ReLU(sum_c w_c * A_c), upsampled to out_size. (B,1,H,W) in [0,1]."""
        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "no activations/gradients captured - run a forward AND backward pass first")
        A = _to_bchw(self.activations)
        G = _to_bchw(self.gradients)
        weights = G.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * A).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=out_size, mode="bilinear", align_corners=False)
        return _normalize(cam)

    def close(self):
        self._handle.remove()


class GradCAM:
    """Single-target-layer Grad-CAM. Use as a context manager."""

    def __init__(self, model, target_layer):
        self.model = model
        self.tap = _Tap(target_layer)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.tap.close()

    def __call__(self, x, class_idx=None):
        """x: (B,3,H,W). Returns (cam (B,1,H,W), logits (B,C), class_idx (B,))."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            if class_idx is None:
                class_idx = logits.argmax(dim=1)
            class_idx = class_idx.view(-1)
            score = logits.gather(1, class_idx.view(-1, 1)).sum()
            score.backward()
        return self.tap.cam(x.shape[-2:]), logits.detach(), class_idx


class TeacherCAM:
    """Grad-CAM for DualBranchFusionTeacher: per-branch CAMs fused by the gate.

    One forward + one backward gives both branch CAMs, because each branch's
    trunk is tapped separately and both receive gradient from the same score.
    """

    def __init__(self, teacher):
        self.model = teacher
        self.layers = teacher.cam_layers()
        self.taps = [_Tap(layer) for layer in self.layers]
        self.branch_names = getattr(teacher, "backbone_names",
                                    tuple("branch%d" % i for i in range(len(self.layers))))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        for t in self.taps:
            t.close()

    def __call__(self, x, class_idx=None):
        """Returns dict: fused, branches (list), gate (B,n), logits, class_idx."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            if class_idx is None:
                class_idx = logits.argmax(dim=1)
            class_idx = class_idx.view(-1)
            logits.gather(1, class_idx.view(-1, 1)).sum().backward()

        cams = [t.cam(x.shape[-2:]) for t in self.taps]          # each (B,1,H,W)
        gate = self.model.gate_weights(x)                         # (B,n)
        fused = sum(gate[:, i].view(-1, 1, 1, 1) * cams[i] for i in range(len(cams)))
        return {
            "fused": _normalize(fused),
            "branches": cams,
            "branch_names": self.branch_names,
            "gate": gate.detach(),
            "logits": logits.detach(),
            "class_idx": class_idx,
        }


# --------------------------------------------------------------------------- #
# rendering + comparison
# --------------------------------------------------------------------------- #
def get_cmap(name="jet"):
    """matplotlib >=3.9 removed cm.get_cmap; this works on both."""
    import matplotlib
    try:
        return matplotlib.colormaps[name]
    except (AttributeError, KeyError):
        from matplotlib import cm
        return cm.get_cmap(name)


def cam_to_heatmap(cam, cmap="jet"):
    """(H,W) in [0,1] -> (H,W,3) RGB float array."""
    return get_cmap(cmap)(np.clip(cam, 0, 1))[..., :3]


def overlay(image_rgb, cam, alpha=0.45, cmap="jet"):
    """image_rgb (H,W,3) float [0,1] + cam (H,W) [0,1] -> blended RGB."""
    heat = cam_to_heatmap(cam, cmap)
    return np.clip((1 - alpha) * image_rgb + alpha * heat, 0, 1)


def cam_cosine(a, b, eps=1e-8):
    """Cosine similarity between two flattened CAMs - 1.0 = same spatial story."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def cam_iou(a, b, q=0.80):
    """IoU of the top-(1-q) hottest regions of two CAMs.

    Cosine says "similar overall shape"; this says "do they actually point at the
    same patch of leaf", which is the question distillation is meant to answer.
    """
    a, b = np.asarray(a), np.asarray(b)
    ma, mb = a >= np.quantile(a, q), b >= np.quantile(b, q)
    union = np.logical_or(ma, mb).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(ma, mb).sum() / union)

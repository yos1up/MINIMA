"""Export MINIMA-LoFTR to two ONNX graphs for mobile inference.

The original LoFTR forward has dynamic shapes inside (mutual-NN match selection
producing a variable number of pairs), which is hostile to mobile ONNX runtimes.
We split the network into two fixed-shape stages and move the dynamic
match-selection step to the CPU caller:

    Stage 1  (image pair) -> (feat_c0, feat_c1, feat_f0, feat_f1)
    --- on the CPU side: dual-softmax + mutual NN + border mask + window crop ---
    Stage 2  (per-match windows + coarse context) -> sub-pixel offset

Stage 1 has fixed input shape (default 1x1x640x640). Stage 2 has a dynamic
match-count axis. Output coordinates are produced by the CPU pipeline; see
``mobile/pipeline.py``.

Run:
    python -m mobile.export_onnx \
        --ckpt weights/minima_loftr.ckpt \
        --img-size 640 \
        --out-dir mobile/onnx
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOFTR_ROOT = os.path.join(PROJECT_ROOT, "third_party", "LoFTR_minima")
if LOFTR_ROOT not in sys.path:
    sys.path.insert(0, LOFTR_ROOT)

from src.loftr import LoFTR, default_cfg  # noqa: E402


def build_loftr(ckpt_path: str, thr: float = 0.2) -> LoFTR:
    cfg = deepcopy(default_cfg)
    filename = os.path.basename(ckpt_path)
    if filename != "outdoor_ds.ckpt":
        cfg["coarse"]["temp_bug_fix"] = True
    cfg["match_coarse"]["thr"] = thr
    model = LoFTR(config=cfg)
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd["state_dict"] if "state_dict" in sd else sd
    model.load_state_dict(sd, strict=True)
    return model.eval()


class Stage1(nn.Module):
    """Backbone + positional encoding + coarse transformer.

    Inputs are two grayscale images shaped ``(1, 1, H, W)`` with values in
    ``[0, 1]``. ``H`` and ``W`` must be multiples of 8.

    Outputs:
      * ``feat_c0``/``feat_c1``  shape ``(1, Hc*Wc, 256)``  — post-transformer
        coarse features (used both for matching and as context in the fine
        stage).
      * ``feat_f0``/``feat_f1``  shape ``(1, 128, Hf, Wf)`` — fine-level
        features for window cropping (Hf = H/2, Wf = W/2).
    """

    def __init__(self, loftr: LoFTR):
        super().__init__()
        self.backbone = loftr.backbone
        self.pos_encoding = loftr.pos_encoding
        self.loftr_coarse = loftr.loftr_coarse

    def forward(self, image0: torch.Tensor, image1: torch.Tensor):
        # Run the CNN backbone on the concatenated batch (matches training path).
        feats_c, feats_f = self.backbone(torch.cat([image0, image1], dim=0))
        feat_c0, feat_c1 = feats_c[:1], feats_c[1:]
        feat_f0, feat_f1 = feats_f[:1], feats_f[1:]

        # Add positional encoding then flatten H*W -> sequence.
        feat_c0 = self.pos_encoding(feat_c0).flatten(2).transpose(1, 2).contiguous()
        feat_c1 = self.pos_encoding(feat_c1).flatten(2).transpose(1, 2).contiguous()

        # Self/cross attention (linear-attention variant in the released ckpt).
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, None, None)
        return feat_c0, feat_c1, feat_f0, feat_f1


class Stage2(nn.Module):
    """Fine transformer + sub-pixel expectation.

    Inputs (M = number of coarse matches, dynamic):
      * ``feat_f0_win`` ``(M, WW, 128)`` — 5x5 fine windows around match0
      * ``feat_f1_win`` ``(M, WW, 128)`` — 5x5 fine windows around match1
      * ``feat_c0_pick`` ``(M, 256)``    — coarse feature at match0 (post-transformer)
      * ``feat_c1_pick`` ``(M, 256)``    — coarse feature at match1

    Output:
      * ``expec_f`` ``(M, 2)`` — sub-pixel offset in normalized coords ``[-1, 1]``.
        Multiply by ``(W//2) * scale`` to get absolute pixel offset for image1.

    A zero match count (M=0) is invalid for ONNX export; the caller must guard
    against that case.
    """

    def __init__(self, loftr: LoFTR):
        super().__init__()
        fine_pre = loftr.fine_preprocess
        self.cat_c_feat = bool(fine_pre.cat_c_feat)
        if self.cat_c_feat:
            self.down_proj = fine_pre.down_proj
            self.merge_feat = fine_pre.merge_feat
        self.loftr_fine = loftr.loftr_fine
        self.W = int(fine_pre.W)
        self.d_model_f = int(fine_pre.d_model_f)

        # Pre-compute the [-1, 1] grid used by spatial expectation (dsnt).
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.W),
            torch.linspace(-1.0, 1.0, self.W),
            indexing="ij",
        )
        grid = torch.stack([xs, ys], dim=-1).reshape(1, self.W * self.W, 2)
        self.register_buffer("grid_normalized", grid, persistent=False)

    def forward(
        self,
        feat_f0_win: torch.Tensor,
        feat_f1_win: torch.Tensor,
        feat_c0_pick: torch.Tensor,
        feat_c1_pick: torch.Tensor,
    ) -> torch.Tensor:
        WW = self.W * self.W

        if self.cat_c_feat:
            # Project coarse features into fine d_model and broadcast across the window,
            # then merge with fine features via the trained linear layer.
            feat_c = self.down_proj(torch.cat([feat_c0_pick, feat_c1_pick], dim=0))  # (2M, d_f)
            feat_c = feat_c.unsqueeze(1).expand(-1, WW, -1)  # (2M, WW, d_f)
            feat_f = torch.cat([feat_f0_win, feat_f1_win], dim=0)  # (2M, WW, d_f)
            merged = self.merge_feat(torch.cat([feat_f, feat_c], dim=-1))  # (2M, WW, d_f)
            feat_f0_win, feat_f1_win = merged.chunk(2, dim=0)

        feat_f0_win, feat_f1_win = self.loftr_fine(feat_f0_win, feat_f1_win)

        # Center pixel of the window in image0 is the anchor; correlate against the window in image1.
        feat_f0_picked = feat_f0_win[:, WW // 2, :]  # (M, d_f)
        sim = torch.einsum("mc,mrc->mr", feat_f0_picked, feat_f1_win)  # (M, WW)
        heatmap = F.softmax(sim / math.sqrt(self.d_model_f), dim=1)  # (M, WW)

        expec = (heatmap.unsqueeze(-1) * self.grid_normalized).sum(dim=1)  # (M, 2)
        return expec


@torch.no_grad()
def export(ckpt: str, out_dir: str, img_size: int, opset: int = 17):
    assert img_size % 8 == 0, "img_size must be a multiple of 8"
    os.makedirs(out_dir, exist_ok=True)

    loftr = build_loftr(ckpt)
    stage1 = Stage1(loftr).eval()
    stage2 = Stage2(loftr).eval()

    Hc = Wc = img_size // 8
    Hf = Wf = img_size // 2
    W = stage2.W  # fine window size (5)
    d_c = 256
    d_f = stage2.d_model_f

    # ---- Stage 1 export ----
    img0 = torch.randn(1, 1, img_size, img_size)
    img1 = torch.randn(1, 1, img_size, img_size)
    stage1_path = os.path.join(out_dir, f"loftr_stage1_{img_size}.onnx")
    torch.onnx.export(
        stage1,
        (img0, img1),
        stage1_path,
        input_names=["image0", "image1"],
        output_names=["feat_c0", "feat_c1", "feat_f0", "feat_f1"],
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"[stage1] saved {stage1_path}")
    print(f"         image0: (1,1,{img_size},{img_size})")
    print(f"         image1: (1,1,{img_size},{img_size})")
    print(f"         feat_c0/1: (1,{Hc * Wc},{d_c})")
    print(f"         feat_f0/1: (1,{d_f},{Hf},{Wf})")

    # ---- Stage 2 export ----
    M = 4  # dummy match count; the dynamic axis lets ORT accept any positive M
    feat_f0_win = torch.randn(M, W * W, d_f)
    feat_f1_win = torch.randn(M, W * W, d_f)
    feat_c0_pick = torch.randn(M, d_c)
    feat_c1_pick = torch.randn(M, d_c)
    stage2_path = os.path.join(out_dir, "loftr_stage2.onnx")
    torch.onnx.export(
        stage2,
        (feat_f0_win, feat_f1_win, feat_c0_pick, feat_c1_pick),
        stage2_path,
        input_names=["feat_f0_win", "feat_f1_win", "feat_c0_pick", "feat_c1_pick"],
        output_names=["expec_f"],
        dynamic_axes={
            "feat_f0_win": {0: "M"},
            "feat_f1_win": {0: "M"},
            "feat_c0_pick": {0: "M"},
            "feat_c1_pick": {0: "M"},
            "expec_f": {0: "M"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"[stage2] saved {stage2_path}")
    print(f"         feat_f*_win: (M,{W * W},{d_f})")
    print(f"         feat_c*_pick: (M,{d_c})")
    print(f"         expec_f: (M,2)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="weights/minima_loftr.ckpt")
    p.add_argument("--out-dir", default="mobile/onnx")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export(args.ckpt, args.out_dir, args.img_size, args.opset)

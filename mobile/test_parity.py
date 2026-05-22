"""End-to-end parity check: PyTorch MINIMA-LoFTR vs the two-stage ONNX pipeline.

Runs the original PyTorch model and the ONNX pipeline on the demo pair, then
reports the per-match coordinate delta and the set agreement of matched cells.
"""
from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOFTR_ROOT = os.path.join(PROJECT_ROOT, "third_party", "LoFTR_minima")
if LOFTR_ROOT not in sys.path:
    sys.path.insert(0, LOFTR_ROOT)

from src.loftr import LoFTR, default_cfg  # noqa: E402

from mobile.export_onnx import build_loftr  # noqa: E402
from mobile.pipeline import LoFTRPipeline, preprocess_image  # noqa: E402


@torch.no_grad()
def run_pytorch(ckpt: str, img0_path: str, img1_path: str, size: int) -> dict:
    model = build_loftr(ckpt).eval()

    img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    t0, sc0, _ = preprocess_image(img0, size)
    t1, sc1, _ = preprocess_image(img1, size)

    batch = {"image0": torch.from_numpy(t0), "image1": torch.from_numpy(t1)}
    model(batch)
    mkpts0 = batch["mkpts0_f"].cpu().numpy() * sc0
    mkpts1 = batch["mkpts1_f"].cpu().numpy() * sc1
    mconf = batch["mconf"].cpu().numpy()
    return {"mkpts0": mkpts0, "mkpts1": mkpts1, "mconf": mconf}


def run_onnx(stage1: str, stage2: str, img0_path: str, img1_path: str, size: int) -> dict:
    pipe = LoFTRPipeline(stage1, stage2, img_size=size)
    img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    r = pipe.match(img0, img1)
    return {"mkpts0": r.mkpts0, "mkpts1": r.mkpts1, "mconf": r.mconf}


def align_by_kpt0(a: dict, b: dict, tol: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Pair up matches by integer-coarse kpts0 (which should be identical)."""
    # mkpts0 of LoFTR is exactly the coarse grid point in image0; matching them by
    # rounding gives a 1:1 alignment as long as both pipelines see the same pairs.
    keys_a = {tuple(np.round(p).astype(int)): i for i, p in enumerate(a["mkpts0"])}
    keys_b = {tuple(np.round(p).astype(int)): i for i, p in enumerate(b["mkpts0"])}
    common = sorted(set(keys_a) & set(keys_b))
    idx_a = np.array([keys_a[k] for k in common], dtype=np.int64)
    idx_b = np.array([keys_b[k] for k in common], dtype=np.int64)
    return idx_a, idx_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="weights/minima_loftr.ckpt")
    ap.add_argument("--stage1", default="mobile/onnx/loftr_stage1_640.onnx")
    ap.add_argument("--stage2", default="mobile/onnx/loftr_stage2.onnx")
    ap.add_argument("--img0", default="demo/vis_test.png")
    ap.add_argument("--img1", default="demo/depth_test.png")
    ap.add_argument("--size", type=int, default=640)
    args = ap.parse_args()

    pt = run_pytorch(args.ckpt, args.img0, args.img1, args.size)
    onx = run_onnx(args.stage1, args.stage2, args.img0, args.img1, args.size)

    print(f"PyTorch matches: {len(pt['mkpts0'])}")
    print(f"ONNX    matches: {len(onx['mkpts0'])}")

    ia, ib = align_by_kpt0(pt, onx)
    print(f"Common matches (by coarse kpt0): {len(ia)} / "
          f"min({len(pt['mkpts0'])},{len(onx['mkpts0'])})")

    if len(ia) == 0:
        print("No common matches; skipping numeric comparison.")
        return

    d0 = np.linalg.norm(pt["mkpts0"][ia] - onx["mkpts0"][ib], axis=1)
    d1 = np.linalg.norm(pt["mkpts1"][ia] - onx["mkpts1"][ib], axis=1)
    dc = np.abs(pt["mconf"][ia] - onx["mconf"][ib])
    print(f"  delta kpts0 (px): mean={d0.mean():.4f} max={d0.max():.4f}")
    print(f"  delta kpts1 (px): mean={d1.mean():.4f} max={d1.max():.4f}")
    print(f"  delta mconf     : mean={dc.mean():.4e} max={dc.max():.4e}")


if __name__ == "__main__":
    main()

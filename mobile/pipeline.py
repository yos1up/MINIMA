"""ONNX Runtime + numpy reference pipeline for MINIMA-LoFTR.

This module is the Python reference implementation of what a Swift / Kotlin /
C++ client would do around the two ONNX graphs produced by ``export_onnx.py``:

  1. preprocess images (grayscale, resize-by-long-side, zero-pad to square)
  2. run stage1.onnx -> feat_c0/c1, feat_f0/f1
  3. compute dual-softmax confidence matrix
  4. mutual-NN match selection with border mask and threshold
  5. crop 5x5 fine windows around each match
  6. run stage2.onnx -> sub-pixel offsets
  7. assemble final mkpts0/mkpts1 in original image pixel coords

The CPU-side ops match LoFTR's ``CoarseMatching.get_coarse_match`` and
``FinePreprocess`` exactly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort


def preprocess_image(
    img: np.ndarray,
    target_size: int = 640,
    df: int = 8,
    pad: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Match ``DataIOWrapper.preprocess_image`` but in pure numpy.

    Returns
    -------
    tensor : (1, 1, H, W) float32 in [0, 1]
    scale  : (sx, sy) original-pixel-per-network-pixel factors
    shape  : (H, W) of returned tensor
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    s = target_size / max(h, w)
    w_new = int(round(w * s))
    h_new = int(round(h * s))
    w_new = (w_new // df) * df
    h_new = (h_new // df) * df
    img_r = cv2.resize(img, (w_new, h_new))
    scale = np.array([w / w_new, h / h_new], dtype=np.float32)

    if pad:
        size = max(h_new, w_new)
        canvas = np.zeros((size, size), dtype=img_r.dtype)
        canvas[:h_new, :w_new] = img_r
        img_r = canvas

    tensor = img_r.astype(np.float32)[None, None] / 255.0
    return tensor, scale, img_r.shape


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _border_mask(Hc: int, Wc: int, border: int) -> np.ndarray:
    """Return a (Hc, Wc) bool mask that is False on the ``border`` outer cells."""
    m = np.ones((Hc, Wc), dtype=bool)
    if border > 0:
        m[:border] = False
        m[-border:] = False
        m[:, :border] = False
        m[:, -border:] = False
    return m


def select_coarse_matches(
    feat_c0: np.ndarray,
    feat_c1: np.ndarray,
    Hc: int,
    Wc: int,
    *,
    thr: float = 0.2,
    border_rm: int = 2,
    temperature: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dual-softmax + mutual NN + border mask + threshold.

    Inputs are stage1 outputs, shape ``(1, L, C)`` with ``L = Hc*Wc``.
    Returns ``(i_ids, j_ids, mconf)`` -- 1D arrays of length M.
    """
    assert feat_c0.shape[1] == Hc * Wc
    f0 = feat_c0[0] / np.sqrt(feat_c0.shape[-1])
    f1 = feat_c1[0] / np.sqrt(feat_c1.shape[-1])
    sim = (f0 @ f1.T) / temperature  # (L, L)

    conf = _softmax(sim, axis=0) * _softmax(sim, axis=1)

    valid = _border_mask(Hc, Wc, border_rm).reshape(-1)
    cross = valid[:, None] & valid[None, :]
    conf = np.where(cross, conf, 0.0)

    row_max = conf.max(axis=1, keepdims=True)
    col_max = conf.max(axis=0, keepdims=True)
    mutual = (conf == row_max) & (conf == col_max) & (conf > thr)

    i_ids, j_ids = np.where(mutual)
    mconf = conf[i_ids, j_ids]
    return i_ids.astype(np.int64), j_ids.astype(np.int64), mconf.astype(np.float32)


def crop_fine_windows(
    feat_f: np.ndarray,
    flat_ids: np.ndarray,
    Hc: int,
    Wc: int,
    W: int = 5,
) -> np.ndarray:
    """Replicate ``F.unfold(kernel=W, stride=stride, padding=W//2)`` + index.

    ``feat_f`` is ``(1, C, Hf, Wf)`` with ``Hf = Hc * stride``. Returns
    ``(M, W*W, C)`` windows for each match.
    """
    _, C, Hf, Wf = feat_f.shape
    stride = Hf // Hc
    pad = W // 2

    padded = np.pad(
        feat_f[0],
        ((0, 0), (pad, pad), (pad, pad)),
        mode="constant",
    )  # (C, Hf+2pad, Wf+2pad)

    h_ids = (flat_ids // Wc) * stride  # (M,) — top-left of the window in padded coords
    w_ids = (flat_ids % Wc) * stride

    # Build per-match (W, W) index grids.
    dh = np.arange(W)
    dw = np.arange(W)
    hh = h_ids[:, None, None] + dh[None, :, None]  # (M, W, 1)
    ww = w_ids[:, None, None] + dw[None, None, :]  # (M, 1, W)
    hh = np.broadcast_to(hh, (len(flat_ids), W, W))
    ww = np.broadcast_to(ww, (len(flat_ids), W, W))

    # Gather: (M, W, W, C) -> (M, W*W, C)
    windows = padded[:, hh, ww]  # (C, M, W, W) via numpy advanced indexing
    windows = np.transpose(windows, (1, 2, 3, 0)).reshape(len(flat_ids), W * W, C)
    return windows.astype(np.float32, copy=False)


@dataclass
class MatchResult:
    mkpts0: np.ndarray  # (M, 2) in original-image pixel coords of image0
    mkpts1: np.ndarray  # (M, 2) in original-image pixel coords of image1
    mconf: np.ndarray   # (M,) coarse-level confidence
    mkpts0_c: np.ndarray  # coarse-only mkpts (no sub-pixel refinement)
    mkpts1_c: np.ndarray


class LoFTRPipeline:
    """Thin orchestrator around the two ONNX graphs."""

    def __init__(
        self,
        stage1_path: str,
        stage2_path: str,
        img_size: int = 640,
        df: int = 8,
        pad: bool = True,
        thr: float = 0.2,
        border_rm: int = 2,
        temperature: float = 0.1,
        providers: Optional[list[str]] = None,
    ):
        providers = providers or ["CPUExecutionProvider"]
        self.s1 = ort.InferenceSession(stage1_path, providers=providers)
        self.s2 = ort.InferenceSession(stage2_path, providers=providers)
        self.img_size = img_size
        self.df = df
        self.pad = pad
        self.thr = thr
        self.border_rm = border_rm
        self.temperature = temperature
        self.W = 5  # fine window size; must match the exported graph

    def match(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
        t0, sc0, (H, W) = preprocess_image(img0, self.img_size, self.df, self.pad)
        t1, sc1, _ = preprocess_image(img1, self.img_size, self.df, self.pad)

        feat_c0, feat_c1, feat_f0, feat_f1 = self.s1.run(
            None, {"image0": t0, "image1": t1}
        )

        Hc, Wc = H // 8, W // 8
        i_ids, j_ids, mconf = select_coarse_matches(
            feat_c0,
            feat_c1,
            Hc,
            Wc,
            thr=self.thr,
            border_rm=self.border_rm,
            temperature=self.temperature,
        )

        coarse_scale = H / Hc  # network-pixels per coarse cell (== 8)
        mkpts0_c = np.stack([i_ids % Wc, i_ids // Wc], axis=1).astype(np.float32) * coarse_scale
        mkpts1_c = np.stack([j_ids % Wc, j_ids // Wc], axis=1).astype(np.float32) * coarse_scale

        if len(i_ids) == 0:
            empty = np.zeros((0, 2), dtype=np.float32)
            return MatchResult(empty, empty, mconf, empty, empty)

        win0 = crop_fine_windows(feat_f0, i_ids, Hc, Wc, self.W)
        win1 = crop_fine_windows(feat_f1, j_ids, Hc, Wc, self.W)
        c0_pick = feat_c0[0, i_ids]
        c1_pick = feat_c1[0, j_ids]

        expec = self.s2.run(
            None,
            {
                "feat_f0_win": win0,
                "feat_f1_win": win1,
                "feat_c0_pick": c0_pick,
                "feat_c1_pick": c1_pick,
            },
        )[0]  # (M, 2)

        fine_scale = H / (H // 2)  # = 2
        mkpts1_f = mkpts1_c + expec * (self.W // 2) * fine_scale

        mkpts0 = mkpts0_c * sc0  # network -> original
        mkpts1 = mkpts1_f * sc1
        return MatchResult(mkpts0, mkpts1, mconf, mkpts0_c * sc0, mkpts1_c * sc1)

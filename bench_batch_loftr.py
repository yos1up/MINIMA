"""Benchmark LoFTR inference for the "1 reference (image0) vs N candidates (image1)" use case.

Compares three strategies on CPU:
  (a) Sequential: N independent model(img0, img1[i]) calls.
  (b) Naive batch: one model() call with image0 and image1 stacked to batch N.
  (c) Backbone-shared batch: run backbone on image0 once and on image1[N] once,
      expand image0 features to N, then run the remaining LoFTR stages as batch N.
"""
import argparse
import sys
import time
import warnings

import cv2
import torch
from einops.einops import rearrange
from types import SimpleNamespace

from load_model import load_model


def _flush():
    sys.stdout.flush()


def call_full(model, batch):
    """Run model.forward; LoFTR writes results into the dict in-place."""
    model(dict(batch))


def call_seq_with_cached_img0(model, image0_single, image1_batch):
    """Sequential per-image1 inference, but backbone(image0) is computed once.

    Useful when batch=N would OOM. Memory stays at batch=1 for the transformer,
    while saving (N-1) image0 backbone passes vs plain sequential.
    """
    feat_c0_1, feat_f0_1 = model.backbone(image0_single)

    N = image1_batch.size(0)
    for i in range(N):
        img1_single = image1_batch[i:i+1]
        data = {"image0": image0_single, "image1": img1_single}
        data["bs"] = 1
        data["hw0_i"] = image0_single.shape[2:]
        data["hw1_i"] = img1_single.shape[2:]

        feat_c1_1, feat_f1_1 = model.backbone(img1_single)
        feat_c0 = feat_c0_1
        feat_f0 = feat_f0_1
        feat_c1 = feat_c1_1
        feat_f1 = feat_f1_1

        data["hw0_c"] = feat_c0.shape[2:]
        data["hw1_c"] = feat_c1.shape[2:]
        data["hw0_f"] = feat_f0.shape[2:]
        data["hw1_f"] = feat_f1.shape[2:]

        feat_c0_enc = rearrange(model.pos_encoding(feat_c0), "n c h w -> n (h w) c")
        feat_c1_enc = rearrange(model.pos_encoding(feat_c1), "n c h w -> n (h w) c")

        mask_c0 = mask_c1 = None
        feat_c0_enc, feat_c1_enc = model.loftr_coarse(feat_c0_enc, feat_c1_enc, mask_c0, mask_c1)
        model.coarse_matching(feat_c0_enc, feat_c1_enc, data, mask_c0=mask_c0, mask_c1=mask_c1)

        feat_f0_unfold, feat_f1_unfold = model.fine_preprocess(
            feat_f0, feat_f1, feat_c0_enc, feat_c1_enc, data
        )
        if feat_f0_unfold.size(0) != 0:
            feat_f0_unfold, feat_f1_unfold = model.loftr_fine(feat_f0_unfold, feat_f1_unfold)
        model.fine_matching(feat_f0_unfold, feat_f1_unfold, data)


def call_shared_img0(model, image0_single, image1_batch):
    """LoFTR forward with image0 having batch=1 and image1 having batch=N.

    Saves backbone work for the duplicated image0 copies. The transformer
    stages still operate at batch=N (image0 features are expanded). Mask is
    not supported in this fast path.
    """
    data = {"image0": image0_single, "image1": image1_batch}
    N = image1_batch.size(0)
    data["bs"] = N
    data["hw0_i"] = image0_single.shape[2:]
    data["hw1_i"] = image1_batch.shape[2:]

    feat_c0_1, feat_f0_1 = model.backbone(image0_single)
    feat_c1, feat_f1 = model.backbone(image1_batch)

    feat_c0 = feat_c0_1.expand(N, -1, -1, -1).contiguous()
    feat_f0 = feat_f0_1.expand(N, -1, -1, -1).contiguous()

    data["hw0_c"] = feat_c0.shape[2:]
    data["hw1_c"] = feat_c1.shape[2:]
    data["hw0_f"] = feat_f0.shape[2:]
    data["hw1_f"] = feat_f1.shape[2:]

    feat_c0 = rearrange(model.pos_encoding(feat_c0), "n c h w -> n (h w) c")
    feat_c1 = rearrange(model.pos_encoding(feat_c1), "n c h w -> n (h w) c")

    mask_c0 = mask_c1 = None
    feat_c0, feat_c1 = model.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
    model.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

    feat_f0_unfold, feat_f1_unfold = model.fine_preprocess(
        feat_f0, feat_f1, feat_c0, feat_c1, data
    )
    if feat_f0_unfold.size(0) != 0:
        feat_f0_unfold, feat_f1_unfold = model.loftr_fine(feat_f0_unfold, feat_f1_unfold)
    model.fine_matching(feat_f0_unfold, feat_f1_unfold, data)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--skip", default="", help="comma-separated: sequential,naive,shared,seqcache")
    parser.add_argument("--verify", action="store_true", help="compare matches across methods for correctness")
    parser.add_argument("--ckpt", default="./weights/minima_loftr.ckpt")
    args = parser.parse_args()
    skip = set(s.strip() for s in args.skip.split(",") if s.strip())

    warnings.simplefilter("ignore")
    margs = SimpleNamespace(ckpt=args.ckpt, thr=0.2)
    m_fn = load_model("loftr", margs)
    wrapper = m_fn.__self__
    model = wrapper.model

    img0 = cv2.imread("./demo/vis_test.png", cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread("./demo/depth_test.png", cv2.IMREAD_GRAYSCALE)
    img0_t, *_ = wrapper.preprocess_image(
        img0, wrapper.device, resize=wrapper.img0_size, df=wrapper.df, padding=wrapper.padding
    )
    img1_t, *_ = wrapper.preprocess_image(
        img1, wrapper.device, resize=wrapper.img1_size, df=wrapper.df, padding=wrapper.padding
    )
    print(f"shape per item: {tuple(img0_t.shape)}; N={args.N}; torch_threads={torch.get_num_threads()}"); _flush()

    # warmup (single pair)
    for _ in range(2):
        call_full(model, {"image0": img0_t, "image1": img1_t})

    def median(times):
        return sorted(times)[len(times) // 2]

    results = {}

    if "sequential" not in skip:
        ts = []
        for r in range(args.repeats):
            t0 = time.time()
            for _ in range(args.N):
                call_full(model, {"image0": img0_t, "image1": img1_t})
            ts.append(time.time() - t0)
            print(f"  (a) sequential   r={r}: {ts[-1]:.2f}s"); _flush()
        results["sequential"] = median(ts)
        print(f"  (a) sequential   N={args.N}: median={results['sequential']:.2f}s  per-pair={results['sequential']/args.N:.3f}s"); _flush()

    img0_b = img0_t.repeat(args.N, 1, 1, 1).contiguous()
    img1_b = img1_t.repeat(args.N, 1, 1, 1).contiguous()

    if "naive" not in skip:
        ts = []
        for r in range(args.repeats):
            t0 = time.time()
            try:
                call_full(model, {"image0": img0_b, "image1": img1_b})
            except Exception as e:
                print(f"  (b) naive batch FAILED: {type(e).__name__}: {e}"); _flush()
                ts.append(float('nan'))
                break
            ts.append(time.time() - t0)
            print(f"  (b) naive batch  r={r}: {ts[-1]:.2f}s"); _flush()
        if ts and ts[0] == ts[0]:  # not nan
            results["naive_batch"] = median(ts)
            print(f"  (b) naive batch  N={args.N}: median={results['naive_batch']:.2f}s  per-pair={results['naive_batch']/args.N:.3f}s"); _flush()

    if "shared" not in skip:
        ts = []
        for r in range(args.repeats):
            t0 = time.time()
            try:
                call_shared_img0(model, img0_t, img1_b)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  (c) shared-img0 FAILED: {type(e).__name__}: {e}"); _flush()
                ts.append(float('nan'))
                break
            ts.append(time.time() - t0)
            print(f"  (c) shared-img0  r={r}: {ts[-1]:.2f}s"); _flush()
        if ts and ts[0] == ts[0]:
            results["shared_img0"] = median(ts)
            print(f"  (c) shared-img0  N={args.N}: median={results['shared_img0']:.2f}s  per-pair={results['shared_img0']/args.N:.3f}s"); _flush()

    if "seqcache" not in skip:
        ts = []
        for r in range(args.repeats):
            t0 = time.time()
            try:
                call_seq_with_cached_img0(model, img0_t, img1_b)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  (d) seq+cache FAILED: {type(e).__name__}: {e}"); _flush()
                ts.append(float('nan'))
                break
            ts.append(time.time() - t0)
            print(f"  (d) seq+cache    r={r}: {ts[-1]:.2f}s"); _flush()
        if ts and ts[0] == ts[0]:
            results["seq_cache_img0"] = median(ts)
            print(f"  (d) seq+cache    N={args.N}: median={results['seq_cache_img0']:.2f}s  per-pair={results['seq_cache_img0']/args.N:.3f}s"); _flush()

    if "sequential" in results:
        print("\nSpeedup vs sequential:"); _flush()
        for k, v in results.items():
            print(f"  {k:14s}: {results['sequential']/v:.2f}x")


if __name__ == "__main__":
    main()

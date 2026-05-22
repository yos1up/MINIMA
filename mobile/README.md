# MINIMA-LoFTR mobile inference

Two-stage ONNX export of `minima_loftr` for iOS/Android via ONNX Runtime Mobile.
Match selection and fine-window cropping are intentionally kept on the CPU side
so both graphs have fully static shapes (stage 2 has one dynamic match-count
axis only).

## Pipeline overview

```
                 ┌──────────────────────────┐
 image0 (1,1,H,W)│  loftr_stage1_<H>.onnx   │ feat_c0/1 (1, Hc*Wc, 256)
 image1 (1,1,H,W)│  backbone + pos_enc +    │ feat_f0/1 (1, 128, Hf, Wf)
                 │  coarse transformer      │
                 └──────────┬───────────────┘
                            ▼
                  CPU (numpy / Swift / Kotlin)
                  ─ dual-softmax similarity
                  ─ mutual-NN + border + threshold
                  ─ crop 5x5 fine windows (unfold-equivalent)
                            ▼
                 ┌──────────────────────────┐
   windows + ctx │  loftr_stage2.onnx       │ expec_f (M, 2)
                 │  fine transformer + dsnt │
                 └──────────────────────────┘
```

`Hc = H/8`, `Hf = H/2`. With the default `H = W = 640`, `Hc*Wc = 6400`.

## Files

- `export_onnx.py` — exports `loftr_stage1_<size>.onnx` and `loftr_stage2.onnx`.
- `pipeline.py` — numpy + ONNX Runtime reference of the full match flow.
- `test_parity.py` — PyTorch vs ONNX numerical parity check.

## Quick start

```bash
# 1. Get weights and init submodule
bash weights/download.sh          # or curl minima_loftr.ckpt directly
git submodule update --init third_party/LoFTR_minima

# 2. Export
python -m mobile.export_onnx \
    --ckpt weights/minima_loftr.ckpt \
    --img-size 640 \
    --out-dir mobile/onnx

# 3. Verify parity vs PyTorch reference
python -m mobile.test_parity
```

Expected output on the demo pair (`demo/vis_test.png` vs `demo/depth_test.png`):

```
PyTorch matches: 598
ONNX    matches: 598
Common matches (by coarse kpt0): 598 / min(598,598)
  delta kpts0 (px): mean=0.0000 max=0.0000
  delta kpts1 (px): mean=0.0000 max=0.0001
  delta mconf     : mean=2.0470e-06 max=1.0401e-05
```

## Sizes (FP32, opset 17)

| graph                 | input shape                      | size  |
| --------------------- | -------------------------------- | ----- |
| `loftr_stage1_640`    | `(1,1,640,640)` x2               | ~56 MB |
| `loftr_stage2`        | `(M,25,128)` x2, `(M,256)` x2    | ~1.6 MB |

## Re-exporting at a different resolution

Stage 1 has a fixed input size — re-run the export with `--img-size 480` or
`--img-size 320` to trade accuracy for speed. Stage 2 is resolution-agnostic
(its only dynamic axis is the match count) and can be reused as-is.

## CPU-side ops (what you implement on the device)

`pipeline.py::select_coarse_matches` is the reference; the math, in plain form:

1. Normalise both coarse features by `sqrt(C) = 16`.
2. `sim = feat_c0 @ feat_c1.T / temperature`, `temperature = 0.1`.
3. `conf = softmax(sim, dim=0) * softmax(sim, dim=1)` (dual-softmax).
4. Zero-out cells within `border_rm = 2` of the coarse-grid border.
5. Keep `(i, j)` where `conf[i, j]` is the max of its row **and** its column
   **and** is above `thr = 0.2`.
6. Coarse coords: `mkpts0_c = (i % Wc, i // Wc) * 8`, similarly for image1.

`pipeline.py::crop_fine_windows` is the unfold-equivalent: for each match `i`,
zero-pad the `feat_f` map by 2 on every side, then read the
`5 x 5 x 128` window starting at `((i // Wc) * 4, (i % Wc) * 4)`.

## Mobile integration notes (ONNX Runtime Mobile)

### iOS

1. Build / install `onnxruntime-objc` (CocoaPods) or use the prebuilt xcframework.
2. Convert each ONNX to `.ort` if you want the slim format:
   `python -m onnxruntime.tools.convert_onnx_models_to_ort mobile/onnx/`.
3. Enable the CoreML execution provider; LoFTR's ops (Conv/MatMul/Softmax/Einsum/
   LayerNorm/ELU) are all supported, but CoreML EP currently falls back to CPU
   for `Einsum` — see "Optimisation" below.
4. Run preprocessing (`vImageScale_Planar8` or Accelerate `vDSP_vsdiv` for the
   `/255.` step) and the dual-softmax loop in Swift/Accelerate.

### Android

1. Add `com.microsoft.onnxruntime:onnxruntime-mobile` to Gradle.
2. Enable the NNAPI EP for stage 1 (`SessionOptions.addNnapi()`). Linear-attention
   `Einsum` ops will fall back to CPU but the conv-heavy backbone is the dominant
   cost and runs well on NNAPI.
3. Stage 2 is tiny — keep it on CPU.

### CPU-side post-processing on device

Either reimplement `select_coarse_matches` and `crop_fine_windows` in Swift /
Kotlin (~80 LOC each), or compile a small C++ helper using XNNPACK / Accelerate
for the matmul. The matmul itself is `(6400, 256) @ (256, 6400) = (6400, 6400)`
single-precision — ~21 GFLOPs, achievable in well under 100 ms with Accelerate
(`cblas_sgemm`) or Eigen on modern phones.

## Optimisation roadmap

The exported graphs are FP32 reference builds. Common next steps:

- **FP16**: `python -m onnxruntime.transformers.float16 --input ... --output ...`
  or feed FP16 inputs and let CoreML / NNAPI run in FP16 internally. Expect
  ~50% latency reduction with negligible accuracy delta.
- **Quantised INT8**: PTQ with `onnxruntime.quantization.quantize_static` using
  ~50 calibration image pairs from MegaDepth. The Linear-Attention `Einsum`
  blocks are quantisation-sensitive; calibrate carefully and validate on
  MegaDepth-1500.
- **Resolution sweep**: `--img-size 480` typically halves stage 1 latency at
  small accuracy cost.
- **Replace `Einsum` with reshape+`MatMul`**: some EPs (e.g. NNAPI) lack a
  fused `Einsum` kernel. If you hit a CPU fallback, rewrite `LinearAttention.forward`
  using `bmm` / `matmul` for better EP coverage. The math is identical.
- **CoreML EP – `padding=2` on F.unfold**: not applicable to the graph (we
  cropped the unfold out to the CPU side); this is one of the main reasons the
  split is structured this way.

## Caveats

- `border_rm` is implemented as zeroing the conf matrix before mutual-NN; the
  PyTorch reference instead masks the 5-D `(B, Hc, Wc, Hc, Wc)` mask. The two
  are numerically equivalent when no padding mask is present (the case for
  square padded input).
- `padding=True` is enabled by default in `pipeline.preprocess_image`. The
  PyTorch evaluation scripts use `PADDING=False` for square images (since they
  are already square); to be consistent with their setup when the image is
  already square, the padding step is a no-op anyway.
- `gt_mask` and the training-only sampling branch of `coarse_matching.py` are
  not part of the mobile pipeline.

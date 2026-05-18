"""
Usage:
    $ uvicorn server:app --reload --port 8000
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import numpy as np
import cv2
import time
import warnings
import logging
from pathlib import Path
from types import SimpleNamespace
import matplotlib.cm as cm
from typing import Dict, Any, List, Optional
import os
import threading
from contextlib import asynccontextmanager
import matplotlib.cm as cm

# 依存ライブラリ (同フォルダに配置されている想定)
from load_model import load_model  # ユーザ既存の実装
from src.utils.plotting import make_matching_figure  # demo.py と同じ可視化関数

try:
    # homography 可視化に必要 (任意)
    import torch
    from kornia.geometry.transform import warp_perspective
except ModuleNotFoundError:
    torch = None  # kornia が無い環境では after_homography はスキップ


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# method -> matcher (from_paths) のキャッシュ。サーバ起動時に preload し、
# それ以降はリクエストごとに再ロードしない。
_MATCHER_CACHE: Dict[str, Any] = {}
_MATCHER_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Preload matcher(s) at startup so request latency excludes model load."""
    preload = os.getenv("PRELOAD_METHODS", "loftr")
    for m in [s.strip() for s in preload.split(",") if s.strip()]:
        try:
            logger.info(f"Preloading matcher: {m}")
            _get_matcher(m)
            logger.info(f"Preloaded matcher: {m}")
        except Exception:
            logger.exception(f"Failed to preload matcher {m}")
    yield


app = FastAPI(title="Relative Pose Estimation API", lifespan=lifespan)

###############################################################################
# Utility helpers
###############################################################################

def _read_image_from_upload(file: UploadFile) -> np.ndarray:
    """Decode an ``UploadFile`` into a BGR ``np.ndarray`` usable by OpenCV."""
    data = np.frombuffer(file.file.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail=f"Cannot decode image from {file.filename}.")
    return img


def _read_image_from_upload_gray(file: UploadFile) -> np.ndarray:
    """Decode an ``UploadFile`` directly as grayscale.

    Matches the path used by /relative_pose, which goes through
    ``cv2.imread(..., IMREAD_GRAYSCALE)`` inside ``DataIOWrapper.from_paths``.
    Using ``cv2.cvtColor(BGR2GRAY)`` after a color decode produces slightly
    different grayscale pixels and shifts LoFTR match counts by ~1-2%.
    """
    data = np.frombuffer(file.file.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail=f"Cannot decode image from {file.filename}.")
    return img


def _make_temp_image(img: np.ndarray, suffix: str = ".png") -> str:
    """Write *img* to a NamedTemporaryFile and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    cv2.imwrite(tmp.name, img)
    return tmp.name


def _default_args_for(method: str) -> SimpleNamespace:
    """Build the argument namespace expected by ``load_model`` using default hyper‑parameters."""
    if method == "xoftr":
        d = {
            "match_threshold": 0.3,
            "fine_threshold": 0.1,
            "ckpt": "./weights/weights_xoftr_640.ckpt",
        }
    elif method == "loftr":
        d = {
            "ckpt": "./weights/minima_loftr.ckpt",
            "thr": 0.2,
        }
    elif method == "sp_lg":
        d = {
            "ckpt": "./weights/minima_lightglue.pth",
        }
    elif method == "roma":
        d = {
            "ckpt": "./weights/minima_roma.pth",
            "ckpt2": "large",
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unknown method: {method}")

    d.update({
        "exp_name": "VisSYN",
        "fig1": "",
        "fig2": "",
        "save_dir": "./demo/",
    })
    return SimpleNamespace(**d)


def _get_matcher(method: str):
    """Return a cached matcher for *method*, loading it on first use."""
    cached = _MATCHER_CACHE.get(method)
    if cached is not None:
        return cached
    with _MATCHER_LOCK:
        cached = _MATCHER_CACHE.get(method)
        if cached is not None:
            return cached
        args = _default_args_for(method)
        logger.info(f"Loading matcher (method={method}, args={args})")
        matcher = load_model(method, args)
        _MATCHER_CACHE[method] = matcher
        return matcher


def _run_matcher(img0_path: str, img1_path: str, method: str) -> Dict[str, Any]:
    """Run cached matcher on two image *paths*."""
    matcher = _get_matcher(method)
    return matcher(img0_path, img1_path)


def _run_seq_cache_img0_loftr(image0: np.ndarray, images1: List[np.ndarray]) -> List[Dict[str, np.ndarray]]:
    """LoFTR "1 reference vs N candidates" with image0's backbone features cached.

    Images may be passed as grayscale (HxW) or BGR (HxWx3); the wrapper's
    preprocess_image handles both. For bit-exact agreement with /relative_pose
    use grayscale (see ``_read_image_from_upload_gray``).

    On CPU this is ~1.5-1.8x faster than calling the matcher N times in a loop,
    keeps memory at batch=1 (vs naive batch=N which OOMs around N=32), and
    produces bit-exact identical matches to the per-pair path.
    """
    if torch is None:
        raise HTTPException(status_code=500, detail="torch not available")
    from einops.einops import rearrange

    matcher_fn = _get_matcher("loftr")
    wrapper = matcher_fn.__self__  # DataIOWrapper
    model = wrapper.model           # LoFTR
    device = wrapper.device

    if wrapper.padding:
        # The fast path skips mask handling; reject padded configs explicitly
        # so we never silently produce wrong results.
        raise HTTPException(status_code=500, detail="seq_cache path requires padding=False in LoFTR config")

    img0_t, scale0, _, _, _ = wrapper.preprocess_image(
        image0, device, resize=wrapper.img0_size, df=wrapper.df, padding=wrapper.padding
    )
    feat_c0_1, feat_f0_1 = model.backbone(img0_t)

    out: List[Dict[str, np.ndarray]] = []
    for image1 in images1:
        img1_t, scale1, _, _, _ = wrapper.preprocess_image(
            image1, device, resize=wrapper.img1_size, df=wrapper.df, padding=wrapper.padding
        )
        feat_c1_1, feat_f1_1 = model.backbone(img1_t)

        data: Dict[str, Any] = {
            "image0": img0_t, "image1": img1_t,
            "bs": 1,
            "hw0_i": img0_t.shape[2:], "hw1_i": img1_t.shape[2:],
            "hw0_c": feat_c0_1.shape[2:], "hw1_c": feat_c1_1.shape[2:],
            "hw0_f": feat_f0_1.shape[2:], "hw1_f": feat_f1_1.shape[2:],
        }
        fc0 = rearrange(model.pos_encoding(feat_c0_1), "n c h w -> n (h w) c")
        fc1 = rearrange(model.pos_encoding(feat_c1_1), "n c h w -> n (h w) c")
        fc0, fc1 = model.loftr_coarse(fc0, fc1, None, None)
        model.coarse_matching(fc0, fc1, data, mask_c0=None, mask_c1=None)
        ff0_u, ff1_u = model.fine_preprocess(feat_f0_1, feat_f1_1, fc0, fc1, data)
        if ff0_u.size(0) != 0:
            ff0_u, ff1_u = model.loftr_fine(ff0_u, ff1_u)
        model.fine_matching(ff0_u, ff1_u, data)

        out.append({
            "mkpts0": data["mkpts0_f"].cpu().numpy() * scale0,
            "mkpts1": data["mkpts1_f"].cpu().numpy() * scale1,
            "mconf": data["mconf"].cpu().numpy(),
        })
    return out

###############################################################################
# Figure saving helpers (demo.py と同等)
###############################################################################

def _save_matching_figure(path: str, img0_rgb: np.ndarray, img1_rgb: np.ndarray,
                          mkpts0: np.ndarray, mkpts1: np.ndarray,
                          inlier_mask: Optional[np.ndarray], color: np.ndarray):
    """Save figure with inlierマッチのみ表示."""
    if inlier_mask is None or len(inlier_mask) == 0:
        return None
    inlier_mask = inlier_mask.astype(bool).squeeze()
    mk0_in = mkpts0[inlier_mask]
    mk1_in = mkpts1[inlier_mask]
    col_in = color[inlier_mask]
    text = [f"Matches:{len(mk0_in)}"]
    make_matching_figure(img0_rgb, img1_rgb, mk0_in, mk1_in, col_in, text=text, path=path, dpi=150)
    return path


def _save_matching_figure_all(path: str, img0_rgb: np.ndarray, img1_rgb: np.ndarray,
                              mkpts0: np.ndarray, mkpts1: np.ndarray, color: np.ndarray):
    """Save figure with 全マッチ表示."""
    text = [f"Matches:{len(mkpts0)}"]
    make_matching_figure(img0_rgb, img1_rgb, mkpts0, mkpts1, color, text=text, path=path, dpi=150)
    return path


def _save_after_homography(path: str, img0_bgr: np.ndarray, H: np.ndarray):
    """Save warped image after homography (requires torch & kornia)."""
    if torch is None or H is None:
        return None
    img_tensor = torch.tensor(cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    H_t = torch.tensor(H, dtype=torch.float32).unsqueeze(0)
    warped = warp_perspective(img_tensor, H_t, img_tensor.shape[2:], align_corners=True)
    out_img = (warped.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    return path


def _generate_and_save_figures(save_dir: str, base_name: str,
                               img0_bgr: np.ndarray, img1_bgr: np.ndarray,
                               mkpts0: np.ndarray, mkpts1: np.ndarray,
                               mconf: np.ndarray, inliers: Optional[np.ndarray],
                               H: Optional[np.ndarray], method: str) -> List[str]:
    """Wrapper to generate three figures like demo.py. Returns list of file paths."""
    os.makedirs(save_dir, exist_ok=True)
    img0_rgb = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB)
    img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)

    # カラーマップ正規化
    if len(mconf) > 0:
        mconf_n = (mconf - mconf.min()) / (mconf.max() - mconf.min() + 1e-5)
    else:
        mconf_n = np.ones(len(mconf))
    color = cm.jet(mconf_n)

    paths = []
    p_inlier = os.path.join(save_dir, f"{base_name}_after_ransac_{method}.jpg")
    ret = _save_matching_figure(p_inlier, img0_rgb, img1_rgb, mkpts0, mkpts1, inliers, color)
    if ret: paths.append(ret)

    p_all = os.path.join(save_dir, f"{base_name}_before_ransac_{method}.jpg")
    paths.append(_save_matching_figure_all(p_all, img0_rgb, img1_rgb, mkpts0, mkpts1, color))

    if H is not None:
        p_h = os.path.join(save_dir, f"{base_name}_after_homography_{method}.jpg")
        ret = _save_after_homography(p_h, img0_bgr, H)
        if ret:
            paths.append(ret)
    return paths

###############################################################################
# API endpoint
###############################################################################

@app.post("/relative_pose")
async def relative_pose(
    fig1: UploadFile = File(..., description="First image"),
    fig2: UploadFile = File(..., description="Second image"),
    method: str = Form("loftr", description="xoftr | sp_lg | loftr | roma"),
):
    """Estimate relative pose and return match coordinates as well."""
    # 1. decode
    try:
        img0 = _read_image_from_upload(fig1)
        img1 = _read_image_from_upload(fig2)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed reading images")
        raise HTTPException(status_code=500, detail=str(e))

    # 2. temporary files (matcher expects paths)
    tmp0 = _make_temp_image(img0)
    tmp1 = _make_temp_image(img1)

    # 3. run matcher (model is loaded once at startup and cached)
    start = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            match_res = _run_matcher(tmp0, tmp1, method)
    except Exception as e:
        logger.exception("Matcher failed")
        raise HTTPException(status_code=500, detail=str(e))
    elapsed = time.time() - start

    mkpts0 = match_res.get("mkpts0", np.empty((0, 2)))
    mkpts1 = match_res.get("mkpts1", np.empty((0, 2)))
    mconf = match_res.get("mconf", np.ones(len(mkpts0)))

    logger.info(f"Method: {method}, Num matches: {len(mkpts0)}")

    # 4. RANSAC homography
    if len(mkpts0) >= 4:
        H, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC)
        num_inliers = int(inliers.sum()) if inliers is not None else 0
        H_out = H.tolist() if H is not None else None
    else:
        num_inliers, H_out = 0, None
        inliers = None
    
    logger.info(f"Method: {method}, Num matches: {len(mkpts0)}, Num inliers: {num_inliers}, Elapsed time: {elapsed:.2f} sec")

    # 5. save figures -------------------------------------------------------
    save_figs = True
    save_dir = Path("./server_debug/")
    if save_figs:
        base_name = f"{Path(fig1.filename).stem}_{Path(fig2.filename).stem}"
        try:
            _generate_and_save_figures(
                save_dir, base_name, img0, img1, mkpts0, mkpts1, mconf, inliers, H, method
            )
        except Exception as e:
            logger.exception("Failed to generate figures")

    # 6. format response (include coordinates)
    response = {
        "method": method,
        "num_matches": int(len(mkpts0)),
        "num_inliers": num_inliers,
        "homography": H_out,
        "elapsed_time_sec": elapsed,
        "mkpts0": mkpts0.tolist(),  # list[list[float, float]]
        "mkpts1": mkpts1.tolist(),
        "inlier_mask": inliers.astype(int).flatten().tolist() if inliers is not None else [],
    }

    return JSONResponse(content=response)

@app.post("/relative_pose_one_to_many")
async def relative_pose_one_to_many(
    fig0: UploadFile = File(..., description="Reference image (image0, used as the anchor)"),
    figs1: List[UploadFile] = File(..., description="Candidate images (image1 list)"),
    method: str = Form("loftr", description="Only 'loftr' is supported in the fast path"),
):
    """Match one reference image (fig0) against N candidate images (figs1).

    For LoFTR this caches image0's backbone features so the per-pair cost is
    ~30-40% lower than calling /relative_pose N times.
    """
    if method != "loftr":
        raise HTTPException(
            status_code=400,
            detail=f"one_to_many fast path only supports method='loftr', got '{method}'. "
                   f"Call /relative_pose N times for other methods."
        )

    try:
        img0_gray = _read_image_from_upload_gray(fig0)
        images1_gray = [_read_image_from_upload_gray(f) for f in figs1]
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed reading images")
        raise HTTPException(status_code=500, detail=str(e))

    if not images1_gray:
        raise HTTPException(status_code=400, detail="figs1 must contain at least one image")

    start = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            match_list = _run_seq_cache_img0_loftr(img0_gray, images1_gray)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Matcher failed")
        raise HTTPException(status_code=500, detail=str(e))
    elapsed = time.time() - start

    results: List[Dict[str, Any]] = []
    for mr in match_list:
        mkpts0, mkpts1, mconf = mr["mkpts0"], mr["mkpts1"], mr["mconf"]
        if len(mkpts0) >= 4:
            H, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC)
            num_inliers = int(inliers.sum()) if inliers is not None else 0
            H_out = H.tolist() if H is not None else None
        else:
            num_inliers, H_out, inliers = 0, None, None
        results.append({
            "num_matches": int(len(mkpts0)),
            "num_inliers": num_inliers,
            "homography": H_out,
            "mkpts0": mkpts0.tolist(),
            "mkpts1": mkpts1.tolist(),
            "inlier_mask": inliers.astype(int).flatten().tolist() if inliers is not None else [],
        })

    N = len(images1_gray)
    logger.info(f"one_to_many method={method} N={N} elapsed={elapsed:.2f}s ({elapsed/N:.3f}s/pair)")

    return JSONResponse(content={
        "method": method,
        "num_candidates": N,
        "elapsed_time_sec": elapsed,
        "results": results,
    })

###############################################################################
# Local dev
###############################################################################

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

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

app = FastAPI(title="Relative Pose Estimation API")

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


def _run_matcher(img0_path: str, img1_path: str, args: SimpleNamespace, method: str) -> Dict[str, Any]:
    """Load matcher once and run it on two image *paths*."""
    print(f"load_model args: {args}")
    matcher = load_model(method, args)
    return matcher(img0_path, img1_path)

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

    # 3. run matcher
    args = _default_args_for(method)
    args.fig1, args.fig2 = tmp0, tmp1

    start = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            match_res = _run_matcher(tmp0, tmp1, args, method)
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

###############################################################################
# Local dev
###############################################################################

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

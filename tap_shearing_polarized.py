import os
import sys
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

# ── Font fix: avoid Chinese glyph warnings on Windows ──────────
if sys.platform.startswith("win"):
    matplotlib.rcParams["font.family"] = "Microsoft YaHei"
elif sys.platform == "darwin":
    matplotlib.rcParams["font.family"] = "Arial"
else:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = r"D:\Document\CF_detection_YOLO V11"
IMAGE_PATH = os.path.join(BASE_DIR, "Raw.png")


# ══════════════════════════════════════════════
# 1. Demosaic + Stokes parameters  (unchanged)
# ══════════════════════════════════════════════

def demosaic(raw_img):
    """Split 2x2 polarization mosaic into four angle channels.
    Layout (Sony IMX250MZR-style sensor):
        0   45
        135 90
    """
    raw = raw_img.astype(np.float32)
    I_0 = raw[0::2, 0::2]
    I_45 = raw[0::2, 1::2]
    I_90 = raw[1::2, 1::2]
    I_135 = raw[1::2, 0::2]
    return I_0, I_45, I_90, I_135


def stokes(I_0, I_45, I_90, I_135):
    """Compute Stokes parameters and derived polarization quantities."""
    S0 = I_0 + I_90
    S1 = I_0 - I_90
    S2 = I_45 - I_135
    S0_safe = np.where(S0 == 0, 1e-5, S0)
    DoLP = np.clip(np.sqrt(S1 ** 2 + S2 ** 2) / S0_safe, 0, 1)
    AoLP = 0.5 * np.arctan2(S2, S1)  # range: (-pi/2, pi/2]
    return S0, S1, S2, DoLP, AoLP


# ══════════════════════════════════════════════
# 2. Tow / background segmentation
# ══════════════════════════════════════════════
#
# This is the step that was MISSING last time. Without it, any
# local/background averaging window that straddles a gap, the frame
# edge, or a foreign object (e.g. that diagonal rod in your photo)
# mixes fiber pixels with non-fiber pixels -- the resulting "local" or
# "background" orientation is meaningless there, and that's exactly
# what lit up your whole result: every gap edge + the rod, nothing
# inside the actual tows.

def build_tow_mask(S0, DoLP, params):
    """Separate carbon-fiber tow pixels from background/gap/foreign objects.

    Default heuristic: Otsu threshold on DoLP (carbon fiber tows are
    strongly, coherently polarizing compared to bare background or most
    non-fiber objects). If you already have a tow/gap mask from your
    Gap-defect pipeline (polarization_defect_detection.py), swap it in
    here instead -- it will be more reliable than this generic fallback.
    """
    dolp_u8 = cv2.normalize(DoLP, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    base_type = cv2.THRESH_BINARY if params['tow_mask_polarity'] == 'high' else cv2.THRESH_BINARY_INV
    _, mask = cv2.threshold(dolp_u8, 0, 255, base_type + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    frac = mask.mean() / 255.0
    print(f"  [mask] tow pixels = {frac * 100:.1f}% of image (polarity='{params['tow_mask_polarity']}')")
    print(f"  [mask] NOTE: check the green contour on panel [1] in the output figure --")
    print(f"         if it doesn't trace your actual tow boundaries, flip 'tow_mask_polarity'")
    print(f"         or plug in your existing Gap-detector mask instead.")
    return mask  # uint8, 255 = tow, 0 = background/gap/other


# ══════════════════════════════════════════════
# 3. Mask-weighted circular statistics on AoLP
# ══════════════════════════════════════════════
#
# Same pi-periodicity fix as before (work in doubled-angle space), but
# now every average is weighted by the tow mask, so background/gap/
# foreign-object pixels never contaminate a tow pixel's local or
# background orientation estimate -- regardless of where the window
# happens to sit relative to an edge.

def circular_stats_masked(AoLP, mask_f, win_size):
    """Mask-weighted local circular mean orientation + coherence R.

    Returns:
        mean_angle : local circular mean orientation (rad)
        R          : coherence / resultant length in [0,1]
        w          : fraction of the window that was actual tow (mask=1).
                     Low w means "this window straddles an edge/gap --
                     don't trust the result here."
    """
    cos2 = np.cos(2 * AoLP) * mask_f
    sin2 = np.sin(2 * AoLP) * mask_f

    w = uniform_filter(mask_f, size=win_size)
    cos2_m = uniform_filter(cos2, size=win_size)
    sin2_m = uniform_filter(sin2, size=win_size)

    w_safe = np.where(w < 1e-6, 1e-6, w)
    cos2_avg = cos2_m / w_safe
    sin2_avg = sin2_m / w_safe

    R = np.clip(np.sqrt(cos2_avg ** 2 + sin2_avg ** 2), 0, 1)
    mean_angle = 0.5 * np.arctan2(sin2_avg, cos2_avg)
    return mean_angle, R, w


def circular_diff(a, b):
    """Signed angular difference a-b, correctly wrapped for pi-periodic AoLP.
    Result always lies in (-pi/2, pi/2]."""
    d = a - b
    return 0.5 * np.arctan2(np.sin(2 * d), np.cos(2 * d))


# ══════════════════════════════════════════════
# 4. Physics-based Twist Feature Map
# ══════════════════════════════════════════════

def compute_twist_map(AoLP, tow_mask, params):
    win_local = params['local_window']
    win_bg = params['background_window']
    mask_f = (tow_mask > 0).astype(np.float32)

    local_angle, R_local, w_local = circular_stats_masked(AoLP, mask_f, win_local)
    bg_angle, R_bg, w_bg = circular_stats_masked(AoLP, mask_f, win_bg)

    dtheta = circular_diff(local_angle, bg_angle)
    dtheta_deg = np.degrees(np.abs(dtheta))

    # A pixel's result is only trustworthy if: it is itself tow, AND
    # both its local and background windows were mostly tow (not
    # straddling a gap / edge / foreign object).
    valid = (mask_f > 0) & (w_local > params['min_valid_fraction']) & (w_bg > params['min_valid_fraction'])

    raw_score = dtheta_deg * R_local
    F_twist = _norm01(np.where(valid, raw_score, 0))

    n_valid = int(valid.sum())
    print(f"  [feature] valid (trustworthy) pixels: {n_valid} ({100 * n_valid / valid.size:.1f}% of image)")
    print(f"  [feature] |dtheta| within valid region: max={dtheta_deg[valid].max() if n_valid else 0:.1f}deg "
          f" mean={dtheta_deg[valid].mean() if n_valid else 0:.1f}deg")
    print(f"  [feature] R_local within valid region : min={R_local[valid].min() if n_valid else 0:.3f} "
          f" mean={R_local[valid].mean() if n_valid else 0:.3f}")
    n_angle = int(((dtheta_deg > params['twist_angle_thresh_deg']) & valid).sum())
    n_both = int(((dtheta_deg > params['twist_angle_thresh_deg']) & (R_local > params['coherence_min']) & valid).sum())
    print(f"  [score]   valid pixels with |dtheta|>{params['twist_angle_thresh_deg']:.0f}deg            = {n_angle}")
    print(f"  [score]   valid pixels with |dtheta|>thr AND R_local>{params['coherence_min']:.2f} = {n_both}")

    return F_twist, dtheta_deg, R_local, R_bg, valid


def _norm01(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


# ══════════════════════════════════════════════
# 5. Mask & Detection Pipeline
# ══════════════════════════════════════════════

def twist_to_mask(dtheta_deg, R_local, valid, angle_thresh, coherence_min, morph_open, morph_close):
    binary = ((dtheta_deg > angle_thresh) & (R_local > coherence_min) & valid).astype(np.uint8) * 255
    if morph_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open,) * 2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    if morph_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_close,) * 2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    return binary


def mask_to_detections(mask, S0, dtheta_deg, min_area, label):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)

        roi_m = np.zeros(S0.shape, dtype=np.uint8)
        cv2.drawContours(roi_m, [cnt], -1, 255, -1)
        mean_s0 = cv2.mean(S0, mask=roi_m)[0]
        mean_dtheta = cv2.mean(dtheta_deg, mask=roi_m)[0]

        detections.append(dict(
            cnt=cnt, x=x, y=y, w=bw, h=bh, label=label,
            mean_s0=mean_s0, mean_dtheta=mean_dtheta
        ))
    return detections


# ══════════════════════════════════════════════
# 6. Visualisation (4 Columns)
# ══════════════════════════════════════════════

def visualize_twist(S0, AoLP, dtheta_deg, R_local, tow_mask, twist_dets, params, save_path=None):
    S0_u8 = cv2.normalize(S0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    orig_bgr = cv2.cvtColor(S0_u8, cv2.COLOR_GRAY2BGR)

    aolp_color = cv2.applyColorMap(cv2.normalize(AoLP, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                                    cv2.COLORMAP_TWILIGHT)
    dtheta_u8 = cv2.normalize(dtheta_deg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dtheta_color = cv2.applyColorMap(dtheta_u8, cv2.COLORMAP_JET)

    canvas = orig_bgr.copy()
    fill_layer = np.zeros_like(canvas)
    for i, d in enumerate(twist_dets, 1):
        cv2.drawContours(fill_layer, [d['cnt']], -1, (0, 165, 255), -1)
        cv2.rectangle(canvas, (d['x'], d['y']), (d['x'] + d['w'], d['y'] + d['h']), (0, 165, 255), 2)
        cv2.putText(canvas, f"{d['label']} #{i} dtheta={d['mean_dtheta']:.1f}deg",
                    (d['x'], max(d['y'] - 5, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 165, 255), 1, cv2.LINE_AA)
    canvas = cv2.addWeighted(fill_layer, 0.35, canvas, 1.0, 0)

    if not twist_dets:
        cv2.putText(canvas, "No Twist detected", (10, canvas.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

    # sanity-check overlay: tow mask boundary on the S0 panel
    s0_with_mask = orig_bgr.copy()
    contours, _ = cv2.findContours(tow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(s0_with_mask, contours, -1, (0, 255, 0), 1)

    fig, axes = plt.subplots(1, 4, figsize=(26, 6))
    fig.suptitle(
        f"Twist Detection Pipeline | local_win={params['local_window']} bg_win={params['background_window']} | "
        f"angle_thr={params['twist_angle_thresh_deg']:.0f}deg coherence_min={params['coherence_min']:.2f} "
        f"min_valid_frac={params['min_valid_fraction']:.2f} | Detected Twist={len(twist_dets)}",
        fontsize=11, y=1.01
    )

    axes[0].imshow(cv2.cvtColor(s0_with_mask, cv2.COLOR_BGR2RGB))
    axes[0].set_title("[1] S0 + tow-mask boundary (green) - CHECK THIS FIRST", fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(aolp_color, cv2.COLOR_BGR2RGB))
    axes[1].set_title("[2] AoLP Map (Orientation)", fontsize=10)
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(dtheta_color, cv2.COLOR_BGR2RGB))
    axes[2].contour(R_local, levels=[params['coherence_min']], colors=['lime'], linewidths=0.8, linestyles=':')
    axes[2].set_title("[3] |dtheta| local-vs-background (Twist Candidates)", fontsize=10)
    axes[2].axis('off')

    axes[3].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axes[3].set_title(f"[4] Twist Result ({len(twist_dets)} found)", fontsize=10, color='orange')
    axes[3].axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[saved] {save_path}")
    plt.show()


# ══════════════════════════════════════════════
# 7. Parameters
# ══════════════════════════════════════════════

PARAMS = dict(
    tow_mask_polarity='high',     # 'high': tow = high DoLP. Flip to 'low' if your tows show LOWER DoLP than background.
    local_window=15,              # local patch window ~ scale of a twist defect / sub-tow width.
                                   # Should be SMALLER than your tow width in pixels.
    background_window=51,         # nominal-orientation window, several x local_window.
                                   # Should still be smaller than the tow's length scale; the mask
                                   # weighting keeps it from leaking into neighboring tows/gaps either way.
    twist_angle_thresh_deg=15.0,  # min |dtheta| (deg) between local & background orientation
    coherence_min=0.30,           # min local coherence R_local required (filters out noise/disorder)
    min_valid_fraction=0.6,       # local/background window must be >=60% real tow pixels to be trusted
    morph_open=3,
    morph_close=7,
    min_area=80,
)

# ══════════════════════════════════════════════
# 8. Main Execution
# ══════════════════════════════════════════════

if __name__ == "__main__":
    raw = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    print(f"[load] shape={raw.shape}  dtype={raw.dtype}")
    I0, I45, I90, I135 = demosaic(raw)
    S0, S1, S2, DoLP, AoLP = stokes(I0, I45, I90, I135)

    print("[compute] tow/background mask ...")
    tow_mask = build_tow_mask(S0, DoLP, PARAMS)

    print("[compute] twist feature map ...")
    F_twist, dtheta_deg, R_local, R_bg, valid = compute_twist_map(AoLP, tow_mask, PARAMS)

    twist_mask = twist_to_mask(dtheta_deg, R_local, valid, PARAMS['twist_angle_thresh_deg'],
                                PARAMS['coherence_min'], PARAMS['morph_open'], PARAMS['morph_close'])
    twist_dets = mask_to_detections(twist_mask, S0, dtheta_deg, PARAMS['min_area'], 'Twist')

    print(f"\n[result] Detected Twist = {len(twist_dets)}")
    for i, d in enumerate(twist_dets, 1):
        print(f"  Twist #{i} -> x={d['x']:4d} y={d['y']:4d} w={d['w']:4d} h={d['h']:4d} | mean|dtheta|={d['mean_dtheta']:.1f} deg")

    save_path = os.path.join(BASE_DIR, "output", "result_twist.png")
    visualize_twist(S0, AoLP, dtheta_deg, R_local, tow_mask, twist_dets, PARAMS, save_path=save_path)
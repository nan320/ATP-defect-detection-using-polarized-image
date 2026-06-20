import os
import sys
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter
BASE_DIR   = r"D:\Document\CF_detection_YOLO V11"
IMAGE_PATH = os.path.join(BASE_DIR, "Raw.png")


# ══════════════════════════════════════════════
# 1. Demosaic + Stokes parameters
# ══════════════════════════════════════════════

def demosaic(raw_img):
    """
    Split 2x2 polarization mosaic into four angle channels.
    Layout: top-left=0deg, top-right=45deg, bottom-right=90deg, bottom-left=135deg
    """
    raw   = raw_img.astype(np.float32)
    I_0   = raw[0::2, 0::2]
    I_45  = raw[0::2, 1::2]
    I_90  = raw[1::2, 1::2]
    I_135 = raw[1::2, 0::2]
    return I_0, I_45, I_90, I_135


def stokes(I_0, I_45, I_90, I_135):
    """
    Compute Stokes parameters and derived polarization quantities.
    """
    S0      = I_0 + I_90
    S1      = I_0 - I_90
    S2      = I_45 - I_135
    S0_safe = np.where(S0 == 0, 1e-5, S0)
    DoLP    = np.clip(np.sqrt(S1**2 + S2**2) / S0_safe, 0, 1)
    AoLP    = 0.5 * np.arctan2(S2, S1)
    return S0, S1, S2, DoLP, AoLP


# ══════════════════════════════════════════════
# 2. Physics-based feature map computation (Updated)
# ══════════════════════════════════════════════

def compute_feature_maps(S0, DoLP, params):
    """
    Compute two score maps encoding physical defect signatures.

    F_gap (Gap confidence) - UPDATED:
      Physics: Gap creates a surface discontinuity, making DoLP drop locally (DoLP valley).
               Meanwhile, S0 will show a sudden variation due to geometric/reflectance changes,
               which could either be brighter (specular reflection) or darker (shadow cavity).
      Formula: F_gap = clip(DoLP_local_mean - DoLP, 0) * abs(S0 - S0_local_mean)

    F_overlap (Overlap confidence):
      Physics: Overlap creates a local step -> extra material raises S0 (S0 bump)
               -> step boundary causes normal discontinuity -> DoLP gradient spike
      Formula: F_overlap = clip(S0 - S0_local_mean, 0) * DoLP_gradient_magnitude
    """
    win      = params['local_window']
    dolp_f32 = DoLP.astype(np.float32)
    S0_f32   = S0.astype(np.float32)

    # Local neighbourhood mean (background reference)
    dolp_mean = uniform_filter(dolp_f32, size=win).astype(np.float32)
    s0_mean   = uniform_filter(S0_f32,   size=win).astype(np.float32)

    # Finding the valley of DOLP and absolute deviation
    dolp_below = np.clip(dolp_mean - dolp_f32, 0, None)
    s0_absolute_deviation = np.abs(S0_f32 - s0_mean)

    # Overlap
    s0_above   = np.clip(S0_f32 - s0_mean, 0, None)       #

    # DoLP space dradient
    gx        = cv2.Sobel(dolp_f32, cv2.CV_32F, 1, 0, ksize=3)
    gy        = cv2.Sobel(dolp_f32, cv2.CV_32F, 0, 1, ksize=3)
    dolp_grad = np.sqrt(gx**2 + gy**2)

    #
    F_gap     = _norm01(dolp_below * s0_absolute_deviation)
    F_overlap = _norm01(s0_above   * dolp_grad)

    print(f"  [feature] dolp_below       : max={dolp_below.max():.4f}  mean={dolp_below.mean():.4f}")
    print(f"  [feature] s0_abs_deviation : max={s0_absolute_deviation.max():.4f}  mean={s0_absolute_deviation.mean():.4f}")
    print(f"  [feature] s0_above         : max={s0_above.max():.4f}  mean={s0_above.mean():.4f}")
    print(f"  [feature] dolp_grad        : max={dolp_grad.max():.4f}  mean={dolp_grad.mean():.4f}")
    print(f"  [score]   F_gap            : max={F_gap.max():.4f}  pixels>{params['gap_score_thresh']:.2f} = {int((F_gap > params['gap_score_thresh']).sum())}")
    print(f"  [score]   F_overlap        : max={F_overlap.max():.4f}  pixels>{params['overlap_score_thresh']:.2f} = {int((F_overlap > params['overlap_score_thresh']).sum())}")

    return F_gap, F_overlap, _norm01(dolp_grad)


def _norm01(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


# ══════════════════════════════════════════════
# 3. Score map -> binary mask -> contours
# ══════════════════════════════════════════════

def score_to_mask(score_map, threshold, morph_open, morph_close):
    binary = (score_map > threshold).astype(np.uint8) * 255
    if morph_open > 0:
        k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open,) * 2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    if morph_close > 0:
        k      = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_close,) * 2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    return binary


def mask_to_detections(mask, S0, DoLP, min_area, label, normal_x, normal_dist):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        cx = x + bw // 2

        if any(abs(cx - rx) < normal_dist and bw < 6 for rx in normal_x):
            continue

        roi_m = np.zeros(S0.shape, dtype=np.uint8)
        cv2.drawContours(roi_m, [cnt], -1, 255, -1)
        mean_s0   = cv2.mean(S0,   mask=roi_m)[0]
        mean_dolp = cv2.mean(DoLP, mask=roi_m)[0]

        detections.append(dict(
            cnt=cnt, x=x, y=y, w=bw, h=bh, label=label,
            mean_s0=mean_s0, mean_dolp=mean_dolp
        ))
    return detections


# ══════════════════════════════════════════════
# 4. Draw single-class result image
# ══════════════════════════════════════════════

def _draw_class(base_bgr, detections, color_bgr, label):
    canvas     = base_bgr.copy()
    fill_layer = np.zeros_like(canvas)

    for i, d in enumerate(detections, 1):
        cv2.drawContours(fill_layer, [d['cnt']], -1, color_bgr, -1)
        cv2.rectangle(canvas, (d['x'], d['y']), (d['x'] + d['w'], d['y'] + d['h']), color_bgr, 2)
        cv2.putText(canvas, f"{d['label']} #{i} S0={d['mean_s0']:.0f} DoLP={d['mean_dolp']:.2f}",
                    (d['x'], max(d['y'] - 5, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color_bgr, 1, cv2.LINE_AA)

    canvas = cv2.addWeighted(fill_layer, 0.38, canvas, 1.0, 0)
    if not detections:
        cv2.putText(canvas, f"No {label} detected", (10, canvas.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_bgr, 2)
    return canvas


# ══════════════════════════════════════════════
# 5. Main visualisation (4 columns)
# ══════════════════════════════════════════════

def visualize(S0, DoLP, F_gap, F_overlap, gap_dets, ovlp_dets, params, save_path=None):
    S0_u8    = cv2.normalize(S0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    orig_bgr = cv2.cvtColor(S0_u8, cv2.COLOR_GRAY2BGR)
    dolp_color = cv2.applyColorMap((DoLP * 255).astype(np.uint8), cv2.COLORMAP_HOT)

    gap_bgr  = _draw_class(orig_bgr, gap_dets,  (0, 0, 255),   'Gap')
    ovlp_bgr = _draw_class(orig_bgr, ovlp_dets, (255, 0, 255), 'Overlap')

    fig, axes = plt.subplots(1, 4, figsize=(26, 6))
    fig.suptitle(
        f"Physics-based Detection | window={params['local_window']} "
        f"gap_thr={params['gap_score_thresh']:.3f} ovlp_thr={params['overlap_score_thresh']:.3f} | "
        f"Gap={len(gap_dets)} Overlap={len(ovlp_dets)}", fontsize=11, y=1.01
    )

    axes[0].imshow(S0_u8, cmap='gray')
    axes[0].set_title("[1] Original (S0 intensity)", fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(dolp_color, cv2.COLOR_BGR2RGB))
    axes[1].contour(F_gap,     levels=[params['gap_score_thresh']], colors=['cyan'],   linewidths=1.0, linestyles='--')
    axes[1].contour(F_overlap, levels=[params['overlap_score_thresh']], colors=['yellow'], linewidths=1.0, linestyles='--')
    axes[1].set_title("[2] DoLP (cyan=Gap score | yellow=Overlap score)", fontsize=9)
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(gap_bgr, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"[3] Gap Only ({len(gap_dets)} detected)\nDoLP dip + S0 deviation", fontsize=10, color='red')
    axes[2].axis('off')

    axes[3].imshow(cv2.cvtColor(ovlp_bgr, cv2.COLOR_BGR2RGB))
    axes[3].set_title(f"[4] Overlap Only ({len(ovlp_dets)} detected)\nS0 bright step + DoLP gradient boundary", fontsize=10, color='purple')
    axes[3].axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[saved] {save_path}")
    plt.show()


# ══════════════════════════════════════════════
# 6. Parameters
# ══════════════════════════════════════════════

PARAMS = dict(
    local_window         = 31,     #
    gap_score_thresh     = 0.1,   # range from 0 to 1
    overlap_score_thresh = 0.10,   # range from 0 to 1
    morph_open           = 5,      # denoise
    morph_close          = 5,      #
    min_area             = 100,    #
    normal_dist          = 8,      #
    peak_distance        = 80,
    peak_prominence      = 15,
)


# ══════════════════════════════════════════════
# 7. Main Execution
# ══════════════════════════════════════════════

if __name__ == "__main__":
    raw = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    print(f"[load] shape={raw.shape}  dtype={raw.dtype}")
    I0, I45, I90, I135     = demosaic(raw)
    S0, S1, S2, DoLP, AoLP = stokes(I0, I45, I90, I135)

    print(f"  S0   range : {S0.min():.1f} ~ {S0.max():.1f}")
    print(f"  DoLP range : {DoLP.min():.4f} ~ {DoLP.max():.4f}")

    print("[compute] feature maps ...")
    F_gap, F_overlap, F_grad = compute_feature_maps(S0, DoLP, PARAMS)

    # finding normal tow standard
    s2_norm  = cv2.normalize(S2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    x_proj   = np.mean(s2_norm, axis=0)
    normal_x, _ = find_peaks(x_proj, distance=PARAMS['peak_distance'], prominence=PARAMS['peak_prominence'])
    print(f"  Normal seam baselines: {len(normal_x)} found  x={normal_x}")

    # Gap extraction
    gap_mask = score_to_mask(F_gap, PARAMS['gap_score_thresh'], PARAMS['morph_open'], PARAMS['morph_close'])
    gap_dets = mask_to_detections(gap_mask, S0, DoLP, PARAMS['min_area'], 'Gap', normal_x, PARAMS['normal_dist'])

    # Overlap extraction
    ovlp_mask = score_to_mask(F_overlap, PARAMS['overlap_score_thresh'], PARAMS['morph_open'], PARAMS['morph_close'])
    ovlp_dets = mask_to_detections(ovlp_mask, S0, DoLP, PARAMS['min_area'], 'Overlap', normal_x, PARAMS['normal_dist'])

    print(f"\n[result] Gap={len(gap_dets)}  Overlap={len(ovlp_dets)}")

    save_path = os.path.join(BASE_DIR, "output", "result_physics.png")
    visualize(S0, DoLP, F_gap, F_overlap, gap_dets, ovlp_dets, PARAMS, save_path=save_path)
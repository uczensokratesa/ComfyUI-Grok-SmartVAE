"""
ltx_chunking_suite/utils/color_utils.py
─────────────────────────────────────────────────────────────────────────────
Pure-function temporal colour matching utilities.
No ComfyUI dependencies. Operates on [T, H, W, 3] float32 tensors.

Used by ltx_video_decoder.py to reduce inter-batch colour flicker during
streaming VAE decode.
"""

from typing import Dict, Optional, Tuple

import torch


# ─────────────────────────────────────────────────────────────────────────────
#  Stats helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_stats(frames: torch.Tensor, n_frames: int) -> Dict:
    """
    Compute per-channel mean and std over the last n_frames of a
    [T, H, W, 3] tensor.  Returns a plain dict — no state.
    """
    n  = min(n_frames, frames.shape[0])
    if n == 0:
        return {"mean": [0.5, 0.5, 0.5], "std": [0.1, 0.1, 0.1]}

    zone = frames[-n:]                          # [n, H, W, 3]
    mean = []
    std  = []
    for c in range(3):
        ch = zone[..., c]
        mean.append(ch.mean().item())
        std.append(max(ch.std().item(), 1e-6))
    return {"mean": mean, "std": std}


def extract_image_stats(img: torch.Tensor) -> Optional[Dict]:
    """
    Compute per-channel mean/std from a ComfyUI IMAGE tensor
    ([1, H, W, C] or [H, W, C] float32).
    Returns None if the input is invalid or empty.
    """
    if img is None:
        return None

    img = img.detach().float().cpu()

    if img.dim() == 4:
        if img.shape[-1] in (3, 4):
            img = img[0, ..., :3]
        elif img.shape[1] in (3, 4):
            img = img.permute(0, 2, 3, 1)[0, ..., :3]
        else:
            return None
    elif img.dim() == 3:
        if img.shape[-1] in (3, 4):
            img = img[..., :3]
        elif img.shape[0] in (3, 4):
            img = img.permute(1, 2, 0)[..., :3]
        else:
            return None
    else:
        return None

    if img.min().item() < 0:
        img = (img + 1.0) / 2.0
    img = torch.clamp(torch.nan_to_num(img), 0.0, 1.0)

    mean, std = [], []
    for c in range(3):
        ch = img[..., c]
        mean.append(ch.mean().item())
        std.append(max(ch.std().item(), 1e-6))
    return {"mean": mean, "std": std}


def update_ema(ema: Dict, new_stats: Dict, momentum: float) -> Dict:
    """
    Exponential moving average update of colour statistics.
    Returns a new dict — does not modify inputs.
    """
    m  = momentum
    om = 1.0 - m
    return {
        "mean": [m * ema["mean"][c] + om * new_stats["mean"][c] for c in range(3)],
        "std":  [m * ema["std"][c]  + om * new_stats["std"][c]  for c in range(3)],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main correction function
# ─────────────────────────────────────────────────────────────────────────────

def apply_temporal_color_match(
    frames: torch.Tensor,
    prev_stats: Optional[Dict],
    ema_stats: Optional[Dict],
    calibration_stats: Optional[Dict],
    correction_strength: float,
    ema_momentum: float,
    fade_frames: int,
) -> Tuple[torch.Tensor, Dict, Dict]:
    """
    Apply temporal colour correction to a batch of frames.

    Strategy
    ────────
    • Batch boundary (first fade_frames frames):
        Reinhard-style per-channel normalisation toward `prev_stats`
        (the tail of the previous batch), weighted by a linear ramp
        0→correction_strength × 0.85.  This removes visible colour
        steps at streaming-decode chunk boundaries.

    • EMA tracking:
        After correction, update an exponential moving average of the
        batch statistics.  This smooths the reference used for the
        *next* batch and damps long-term drift.

    • Calibration:
        When calibration_stats is provided (from a reference image),
        it seeds the initial EMA so the whole video is anchored to
        that colour grade.

    Parameters
    ──────────
    frames             : [T, H, W, 3] float32 in [0, 1]
    prev_stats         : stats dict from the tail of the previous batch
                         (None on the very first batch)
    ema_stats          : running EMA dict (None on the very first batch)
    calibration_stats  : stats from a reference image (optional)
    correction_strength: max blend weight toward the reference (0–1)
    ema_momentum       : EMA decay (0.7–0.98; higher = slower adaptation)
    fade_frames        : number of frames at the batch head to correct

    Returns
    ───────
    corrected_frames   : [T, H, W, 3] corrected tensor
    updated_ema        : new EMA stats dict for the next call
    curr_stats         : stats of the corrected batch tail (→ next prev_stats)
    """
    # Initialise EMA on first call
    if ema_stats is None:
        ema_stats = (
            {k: list(v) for k, v in calibration_stats.items()}
            if calibration_stats is not None
            else {"mean": [0.5, 0.5, 0.5], "std": [0.1, 0.1, 0.1]}
        )

    corrected = frames.clone()

    # ── Boundary correction (ramp over fade_frames) ──────────────────
    if prev_stats is not None and fade_frames > 0 and correction_strength > 0:
        N = min(fade_frames, corrected.shape[0])
        for i in range(N):
            ramp            = (N - i) / N          # 1 → 0 linear
            local_strength  = correction_strength * ramp * 0.85
            for c in range(3):
                ch      = corrected[i, :, :, c]
                curr_m  = ch.mean()
                curr_s  = max(ch.std().item(), 1e-6)
                tgt_m   = prev_stats["mean"][c]
                tgt_s   = prev_stats["std"][c]

                # Reinhard transfer: normalise to target distribution
                normalised = (ch - curr_m) * (tgt_s / curr_s) + tgt_m
                corrected[i, :, :, c] = (
                    (1.0 - local_strength) * ch + local_strength * normalised
                )

    corrected = torch.clamp(corrected, 0.0, 1.0)

    # ── Update EMA with corrected batch stats ─────────────────────────
    curr_stats  = extract_stats(corrected, fade_frames)
    updated_ema = update_ema(ema_stats, curr_stats, ema_momentum)

    return corrected, updated_ema, curr_stats

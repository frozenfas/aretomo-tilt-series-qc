"""Geometry helpers for frame overlap and rectangle corners."""

import numpy as np


def compute_overlap(tx, ty, w, h, rot_deg=0.0):
    """
    % overlap of a frame shifted by (tx, ty) with the reference frame.

    TX and TY in the AreTomo .aln file are in the *aligned* coordinate system
    (i.e. after in-plane rotation by ROT degrees).  To compare the shift
    against the original image dimensions (W × H), the displacement must be
    rotated back to the local frame:

        local_x = cos(ROT)*TX + sin(ROT)*TY   → compare vs W
        local_y = −sin(ROT)*TX + cos(ROT)*TY  → compare vs H

    For ROT ≈ 85° (typical AreTomo tilt axis) the dominant terms are
    local_x ≈ TY and local_y ≈ −TX, so naïvely comparing TX vs W and TY vs H
    gives the wrong answer.

    Parameters
    ----------
    tx, ty   : floats   in-plane translation in the *aligned* frame (px)
    w, h     : floats   original image width and height (px)
    rot_deg  : float    in-plane rotation angle ROT from the .aln file (degrees)
    """
    if rot_deg != 0.0:
        a  = np.radians(rot_deg)
        ca, sa = float(np.cos(a)), float(np.sin(a))
        tx, ty = ca * tx + sa * ty, -sa * tx + ca * ty
    ox = max(0.0, w - abs(tx)) / w
    oy = max(0.0, h - abs(ty)) / h
    return float(ox * oy * 100.0)


def rotated_rect_corners(cx, cy, w, h, angle_deg):
    """
    Return the 4 corners of a W×H rectangle centred at (cx, cy),
    rotated in-plane by angle_deg.
    """
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    hw, hh = w / 2.0, h / 2.0
    local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return [(cx + ca * x - sa * y, cy + sa * x + ca * y) for x, y in local]

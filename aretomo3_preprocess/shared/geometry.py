"""Geometry helpers for frame overlap and rectangle corners."""

import numpy as np


def compute_overlap(tx, ty, w, h):
    """
    % overlap of a frame shifted by (tx, ty) with the reference frame.

    All frames within a tilt series share the same ROT value, so the
    relative displacement between any frame and the reference is purely
    translational in the aligned coordinate system.  The overlap of two
    identically-rotated, same-sized rectangles separated by (tx, ty) is:

        overlap_x = max(0, W - |tx|) / W
        overlap_y = max(0, H - |ty|) / H
        % overlap  = overlap_x * overlap_y * 100
    """
    ox = max(0.0, w - abs(tx)) / w
    oy = max(0.0, h - abs(ty)) / h
    return ox * oy * 100.0


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

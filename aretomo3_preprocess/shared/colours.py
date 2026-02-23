"""Colour maps and helpers for overlap and CTF resolution plots."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Overlap colourmap: red (0%) → green (100%)
OVL_CMAP = plt.cm.RdYlGn
OVL_NORM = plt.Normalize(vmin=0, vmax=100)


def _ovl_colour(overlap_pct):
    return OVL_CMAP(OVL_NORM(overlap_pct))


def _ovl_sm():
    sm = plt.cm.ScalarMappable(cmap=OVL_CMAP, norm=OVL_NORM)
    sm.set_array([])
    return sm


# Resolution colourmap: green (good/low Å) → red (poor/high Å)
# fit_spacing_A: lower = better resolution → we want low values to be green
# Using RdYlGn_r: value=0 → green, value=1 → red; normalise so low Å → 0
RES_CMAP = plt.cm.RdYlGn_r


def _res_sm(vmin, vmax):
    sm = plt.cm.ScalarMappable(cmap=RES_CMAP,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    return sm

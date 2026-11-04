# SPDX-License-Identifier: MIT
"""Our own 5-tap bicubic phase table (513 x float4: left weight, middle weight, right weight, middle
offset), the role of kBicubicHwPhaseWeightsGlobal in rr2_post.

Derivation (docs/FASE3C.md, section 7): for phase f = k/512 take the Catmull-Rom weights c0..c3 of
texels -1, 0, 1, 2. The middle tap is one bilinear fetch between texels 0 and 1 at offset
o = round(256 * c2 / (c1 + c2)) / 256 (the hardware keeps 8 fractional bits). With o fixed, the three
weights (w0, w12, w3) minimise sum((r - c)^2) over the realised texel weights
r = (w0, w12 (1 - o), w12 o, w3) subject to w0 + w12 + w3 = 1. Closed form, in exact rationals, then
one rounding to fp32. Where the quantised offset hits 0 or 1 (k = 0, 1, 511) the row is the plain
middle texel (0, 1, 0) with the offset kept at most 255/256; k = 512 is (0, 1, 0, 1).

Usage: python gpu/tools/bicubic_table.py OUT.f32           write the table
       python gpu/tools/bicubic_table.py --check            compare with the table of the NVIDIA DLL
                                                           (read at run time through ref-impl, not stored)
"""
import os, sys
from fractions import Fraction as Fr
import numpy as np


def row(k):
    f = Fr(k, 512)
    c = [f * (Fr(-1, 2) + f * (1 - f / 2)), 1 + f * f * (Fr(-5, 2) + Fr(3, 2) * f),
         f * (Fr(1, 2) + f * (2 - Fr(3, 2) * f)), f * f * (Fr(-1, 2) + f / 2)]
    oi = int(c[2] / (c[1] + c[2]) * 256 + Fr(1, 2))
    if k == 512:
        return [0, 1, 0, 1]
    if oi in (0, 256):    # the middle tap degenerates: plain middle texel, offset kept below 1
        return [0, 1, 0, Fr(min(oi, 255), 256)]
    o = Fr(oi, 256)
    q = (1 - o) ** 2 + o ** 2
    p = c[1] * (1 - o) + c[2] * o
    lam = (c[1] + c[2] - p / q) / (2 + 1 / q)
    return [c[0] + lam, (lam + p) / q, c[3] + lam, o]


def table():
    # float(Fraction) is correctly rounded to fp64; the second rounding to fp32 never sees a tie here
    return np.array([[float(v) for v in row(k)] for k in range(513)], np.float64).astype(np.float32)


def check():
    here = os.path.dirname(os.path.abspath(__file__))
    ref = os.environ.get("OPENNR_REFIMPL", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(here))), "OpenNR-Denoiser", "ref-impl"))
    sys.path.insert(0, ref)
    from kernels import post
    nv = post.bicubic_table().astype(np.float32)
    ours = table()
    k = np.arange(513)
    edge = (k <= 16) | (k >= 496)
    eq = (ours == nv).all(1)
    ulp = np.abs(ours.view(np.int32).astype(np.int64) - nv.view(np.int32))
    print("rows bit-equal: %d of 513" % eq.sum())
    print("offsets equal: %d of 513" % (ours[:, 3] == nv[:, 3]).sum())
    print("rows 17..495: %d of 479 equal, max %d ulp" % (eq[~edge].sum(), ulp[~edge].max()))
    print("edge rows (k <= 16, k >= 496): %d of 34 equal, max |diff| %.3g" % (eq[edge].sum(), np.abs(ours - nv)[edge].max()))
    print("max |diff| overall %.3g" % np.abs(ours - nv).max())


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        check()
    else:
        table().tofile(sys.argv[1])
        print("wrote", sys.argv[1])

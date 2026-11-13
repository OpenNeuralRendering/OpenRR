// SPDX-License-Identifier: MIT
// Single elements of the 16x16 fp16 fragments, wave32 (lane l, h = l / 16). The kernels name elements in "row order":
// 8 values per lane, rows 8 h .. 8 h + 7 of column l % 16. On RDNA4 an accumulator and a B operand hold exactly that, and
// an A operand its transpose, so a copy element by element moves a whole matrix.
// RDNA3 / RDNA 3.5 (ONR_GFX11; rr_probe layout on an RX 7900 XTX): accumulator element j = row 2 j + h, column l % 16;
// A and B hold 16 elements per lane, the same in both lane halves: A element i = row l % 16, column i; B element i =
// row i, column l % 16. Whatever a lane lacks is in lane l ^ 16, one subgroupShuffleXor(16) per packed pair.
//   CPB(b, h)    accumulator -> B operand of the same matrix
//   CPA(a, h)    accumulator -> A operand of its transpose (on RDNA3 the same moves as CPB)
//   ROWS8(t, m)  accumulator -> t[8] in row order (RDNA4: t is a copy of the accumulator)
//   PUTB(b, t)   row order -> B operand; PUTH(m, t) row order -> accumulator
//   ROWS8B(t, b) B operand -> t[8] in row order; CPAB(a, b) B operand -> A operand of its transpose (RDNA3: no shuffle)
//   ROWV(x) / RT(x) / ROWH(x) / ROWB(x): build x in row order (in place on RDNA4), then move it into the fragment
//   SM_SUM / SM_E2 / SM_SUMR: the softmax row sum (below)
// LDS_COPY: all of it through a 16x16 tile per subgroup in the LDS (FRAG_SCR, after the body's), on any layout: the
// correctness reference of ONR_GFX11. Every macro exchanges values across lanes: call it where the whole subgroup runs.
#ifndef ONR_GFX11
#define ONR_GFX11 0
#endif
#if ONR_GFX11 && defined(LDS_COPY)
#error "ONR_GFX11 or LDS_COPY, not both"
#endif

#if defined(LDS_COPY)
#define FRAG_ROWS 1
#define FRB (FRAG_SCR + 256u * gl_SubgroupID)
#define LDS_T(m) { SG_SYNC(); coopMatStore(m, lds, FRB, 16u, ROW); SG_SYNC(); }
#define CPA(a, h) { LDS_T(h) coopMatLoad(a, lds, FRB, 16u, COL); }
#define CPB(b, h) { LDS_T(h) coopMatLoad(b, lds, FRB, 16u, ROW); }
#define LDS_E(i) lds[FRB + (8u * (gl_SubgroupInvocationID >> 4) + uint(i)) * 16u + (gl_SubgroupInvocationID & 15u)]
#define ROWS8(t, m) float16_t t[8]; { MH m_ = m; LDS_T(m_) for (int i_ = 0; i_ < 8; ++i_) t[i_] = LDS_E(i_); }
#define PUT8(m, t) { SG_SYNC(); for (int i_ = 0; i_ < 8; ++i_) LDS_E(i_) = t[i_]; SG_SYNC(); coopMatLoad(m, lds, FRB, 16u, ROW); }
#define PUTB(b, t) PUT8(b, t)
#define PUTH(m, t) PUT8(m, t)
#define ROWS8B(t, b) float16_t t[8]; { LDS_T(b) for (int i_ = 0; i_ < 8; ++i_) t[i_] = LDS_E(i_); }
#define CPAB(a, b) { LDS_T(b) coopMatLoad(a, lds, FRB, 16u, COL); }
#elif ONR_GFX11
#define FRAG_ROWS 1
#define FRAG_HI ((gl_SubgroupInvocationID & 16u) != 0u)
#define X16(v) unpackFloat2x16(subgroupShuffleXor(packFloat2x16(v), 16u))
// elements 2 k, 2 k + 1 are rows 4 k + h, 4 k + 2 + h; the other parity is lane l ^ 16's
#define CPB(b, h) { [[unroll]] for (int k_ = 0; k_ < 4; ++k_) { \
        f16vec2 m_ = f16vec2(h[2 * k_], h[2 * k_ + 1]), o_ = X16(m_); \
        f16vec2 ev_ = FRAG_HI ? o_ : m_, od_ = FRAG_HI ? m_ : o_; \
        b[4 * k_] = ev_.x, b[4 * k_ + 1] = od_.x, b[4 * k_ + 2] = ev_.y, b[4 * k_ + 3] = od_.y; } }
#define CPA(a, h) CPB(a, h)
// half 0 keeps its rows 0, 2 .. 6 (elements 0..3) and takes rows 1 .. 7 from half 1's elements 0..3; half 1 keeps rows
// 9 .. 15 (elements 4..7) and takes rows 8 .. 14 from half 0's elements 4..7
#define ROWS8(t, m) float16_t t[8]; { MH m_ = m; [[unroll]] for (int q_ = 0; q_ < 2; ++q_) { \
        f16vec2 lo_ = f16vec2(m_[2 * q_], m_[2 * q_ + 1]), up_ = f16vec2(m_[4 + 2 * q_], m_[5 + 2 * q_]); \
        f16vec2 r_ = X16(FRAG_HI ? lo_ : up_), kp_ = FRAG_HI ? up_ : lo_; \
        f16vec2 ev_ = FRAG_HI ? r_ : kp_, od_ = FRAG_HI ? kp_ : r_; \
        t[4 * q_] = ev_.x, t[4 * q_ + 1] = od_.x, t[4 * q_ + 2] = ev_.y, t[4 * q_ + 3] = od_.y; } }
// the inverse of ROWS8
#define PUTH(m, t) { [[unroll]] for (int q_ = 0; q_ < 2; ++q_) { \
        f16vec2 ev_ = f16vec2(t[4 * q_], t[4 * q_ + 2]), od_ = f16vec2(t[4 * q_ + 1], t[4 * q_ + 3]); \
        f16vec2 r_ = X16(FRAG_HI ? ev_ : od_), kp_ = FRAG_HI ? od_ : ev_; \
        f16vec2 lo_ = FRAG_HI ? r_ : kp_, up_ = FRAG_HI ? kp_ : r_; \
        m[2 * q_] = lo_.x, m[2 * q_ + 1] = lo_.y, m[4 + 2 * q_] = up_.x, m[5 + 2 * q_] = up_.y; } }
// rows 0..7 from half 0, 8..15 from half 1
#define PUTB(b, t) { [[unroll]] for (int q_ = 0; q_ < 4; ++q_) { \
        f16vec2 m_ = f16vec2(t[2 * q_], t[2 * q_ + 1]), o_ = X16(m_); \
        f16vec2 lo_ = FRAG_HI ? o_ : m_, up_ = FRAG_HI ? m_ : o_; \
        b[2 * q_] = lo_.x, b[2 * q_ + 1] = lo_.y, b[8 + 2 * q_] = up_.x, b[9 + 2 * q_] = up_.y; } }
// a B operand holds the whole column: half h takes rows 8 h .. 8 h + 7; as A of the transpose it is the same registers
#define ROWS8B(t, b) float16_t t[8]; { [[unroll]] for (int q_ = 0; q_ < 4; ++q_) { \
        f16vec2 v_ = FRAG_HI ? f16vec2(b[8 + 2 * q_], b[9 + 2 * q_]) : f16vec2(b[2 * q_], b[2 * q_ + 1]); t[2 * q_] = v_.x, t[2 * q_ + 1] = v_.y; } }
#define CPAB(a, b) { [[unroll]] for (int i_ = 0; i_ < 16; ++i_) a[i_] = b[i_]; }
#else
#define FRAG_ROWS 0
#define CPA(a, h) { for (int i_ = 0; i_ < h.length(); ++i_) a[i_] = h[i_]; }
#define CPB(b, h) { for (int i_ = 0; i_ < h.length(); ++i_) b[i_] = h[i_]; }
#define ROWS8(t, m) MH t = m;
#define PUTB(b, t) { for (int i_ = 0; i_ < 8; ++i_) b[i_] = t[i_]; }
#define PUTH(m, t) { for (int i_ = 0; i_ < 8; ++i_) m[i_] = t[i_]; }
#define ROWS8B(t, b) float16_t t[8]; { for (int i_ = 0; i_ < 8; ++i_) t[i_] = b[i_]; }
#define CPAB(a, b) { for (int i_ = 0; i_ < b.length(); ++i_) a[i_] = b[i_]; }
#endif

#if FRAG_ROWS
#define ROWV(x) float16_t x##_r[8];
#define RT(x) x##_r
#define ROWH(x) PUTH(x, x##_r)
#define ROWB(x) PUTB(x, x##_r)
#else
#define ROWV(x)
#define RT(x) x
#define ROWH(x)
#define ROWB(x)
#endif

// Softmax row sum of the transposed bodies over the key tiles p0..p3 (S^T [64 keys][16 own], exponentiated), fp16 in this
// order: per key class e = key % 8 over the groups a = key / 8, ((a0 + a2) + (a4 + a6)) + ((a1 + a3) + (a5 + a7)); then
// ((c0 + c2) + (c4 + c6)) + ((c1 + c3) + (c5 + c7)) over the classes. In row order lane half h holds groups 2 kt + h of
// every class, so e2 (the even or odd groups per class pair) meets the other half's once; SM_E2 makes e2 from p0..p3.
#define SM_E2_ROWS(e2, p0, p1, p2, p3) f16vec2 e2[4]; \
            ROWS8(g0_, p0) ROWS8(g1_, p1) ROWS8(g2_, p2) ROWS8(g3_, p3) \
            [[unroll]] for (int e = 0; e < 8; e += 2) { \
                precise f16vec2 sm = (f16vec2(g0_[e], g0_[e + 1]) + f16vec2(g1_[e], g1_[e + 1])) + (f16vec2(g2_[e], g2_[e + 1]) + f16vec2(g3_[e], g3_[e + 1])); \
                e2[e >> 1] = sm; \
            }
#define SM_PC2(sum, e2) \
            f16vec2 pc2[4]; \
            [[unroll]] for (int i = 0; i < 4; ++i) { \
                f16vec2 o = unpackFloat2x16(subgroupShuffleXor(packFloat2x16(e2[i]), 16u)); \
                precise f16vec2 sp = e2[i] + o; \
                pc2[i] = sp; \
            } \
            precise f16vec2 b2_ = (pc2[0] + pc2[1]) + (pc2[2] + pc2[3]); \
            precise float16_t sum = b2_.x + b2_.y;
#if ONR_GFX11
// a lane holds keys 16 kt + 2 j + h: element m of a tile is group 2 kt, element m + 4 group 2 kt + 1, class 2 m + h, so
// each lane sums 4 whole classes; the two halves' class sums meet once. Pairs (m, m + 2) give (c_m + c_m+1, c_m+2 + c_m+3).
#define SM_SUM(sum, e2, p0, p1, p2, p3) precise float16_t sum; { f16vec2 cl_[2]; \
            [[unroll]] for (int m_ = 0; m_ < 2; ++m_) { \
                precise f16vec2 se_ = (f16vec2(p0[m_], p0[m_ + 2]) + f16vec2(p1[m_], p1[m_ + 2])) + (f16vec2(p2[m_], p2[m_ + 2]) + f16vec2(p3[m_], p3[m_ + 2])); \
                precise f16vec2 so_ = (f16vec2(p0[m_ + 4], p0[m_ + 6]) + f16vec2(p1[m_ + 4], p1[m_ + 6])) + (f16vec2(p2[m_ + 4], p2[m_ + 6]) + f16vec2(p3[m_ + 4], p3[m_ + 6])); \
                precise f16vec2 c_ = se_ + so_; \
                cl_[m_] = c_; } \
            precise f16vec2 w_ = cl_[0] + cl_[1]; \
            precise float16_t x_ = w_.x + w_.y; \
            float16_t o_ = X16(f16vec2(x_, 0.0hf)).x; \
            sum = x_ + o_; }
#define SM_E2(e2, p0, p1, p2, p3)
#define SM_SUMR(sum, e2, p0, p1, p2, p3) SM_SUM(sum, e2, p0, p1, p2, p3)
#elif FRAG_ROWS
#define SM_SUM(sum, e2, p0, p1, p2, p3) SM_E2_ROWS(e2r_, p0, p1, p2, p3) SM_PC2(sum, e2r_)
#define SM_E2(e2, p0, p1, p2, p3) SM_E2_ROWS(e2, p0, p1, p2, p3)
#define SM_SUMR(sum, e2, p0, p1, p2, p3) SM_PC2(sum, e2)
#else
// e2 is already per row order; the tiles are read in place
#define SM_SUM(sum, e2, p0, p1, p2, p3) SM_PC2(sum, e2)
#define SM_E2(e2, p0, p1, p2, p3) f16vec2 e2[4]; \
            [[unroll]] for (int e = 0; e < 8; e += 2) { \
                precise f16vec2 sm = (f16vec2(p0[e], p0[e + 1]) + f16vec2(p1[e], p1[e + 1])) + (f16vec2(p2[e], p2[e + 1]) + f16vec2(p3[e], p3[e + 1])); \
                e2[e >> 1] = sm; \
            }
#define SM_SUMR(sum, e2, p0, p1, p2, p3) SM_PC2(sum, e2)
#endif

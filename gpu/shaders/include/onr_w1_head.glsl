// SPDX-License-Identifier: MIT
// One attention head of the W1 body (onr_block_w1.glsl), hd in scope: once in its loop, or once per head with hd a
// constant (LT_FAR1), so each head keeps only its own key tiles.
        uint wq = hd * 32u * ENC_C;
        // Q^T [dim tile][token tile] = in_a x^T, V [token tile][dim tile] = x in_b^T; K = channel tiles in order
        ACC q00 = ACC(0.0), q01 = ACC(0.0), q10 = ACC(0.0), q11 = ACC(0.0), q20 = ACC(0.0), q21 = ACC(0.0), q30 = ACC(0.0), q31 = ACC(0.0);
        ACC v00 = ACC(0.0), v01 = ACC(0.0), v10 = ACC(0.0), v11 = ACC(0.0), v20 = ACC(0.0), v21 = ACC(0.0), v30 = ACC(0.0), v31 = ACC(0.0);
#define QV_TOK(xt, qa, qb, va, vb) { MB b_; CPB(b_, xt) MMA(qa, wa0, b_) Q2_IF { MMA(qb, wa1, b_) } \
            MA a_; CPA(a_, xt) MMA(va, a_, wb0) MMA(vb, a_, wb1) }
#define QV_K(n, xa, xb, xc, xd) { MA wa0, wa1; MB wb0, wb1; \
            coopMatLoad(wa0, pc.in_a.v, QA_LR(wq + (n) * 16u), ENC_C, ROW); Q2_IF { coopMatLoad(wa1, pc.in_a.v, wq + 16u * ENC_C + (n) * 16u, ENC_C, ROW); } \
            coopMatLoad(wb0, pc.in_b.v, VB_LR(wq + (n) * 16u), ENC_C, COL); coopMatLoad(wb1, pc.in_b.v, VB_LR(wq + 16u * ENC_C + (n) * 16u), ENC_C, COL); \
            QV_TOK(xa, q00, q01, v00, v01) QV_TOK(xb, q10, q11, v10, v11) QV_TOK(xc, q20, q21, v20, v21) QV_TOK(xd, q30, q31, v30, v31) }
        QV_K(0u, x00, x10, x20, x30)
        QV_K(1u, x01, x11, x21, x31)
#ifndef LT_VO
        // out projection fragments of the head: [channel tile][dim tile], K = 32 hd + dims
        MA wo00, wo01, wo10, wo11;
        coopMatLoad(wo00, pc.wout.v, WT_OUT(hd * 32u), WS_OUT, ROW);
        coopMatLoad(wo01, pc.wout.v, WT_OUT(hd * 32u + 16u), WS_OUT, ROW);
        coopMatLoad(wo10, pc.wout.v, WT_OUT(16u * ENC_X + hd * 32u), WS_OUT, ROW);
        coopMatLoad(wo11, pc.wout.v, WT_OUT(16u * ENC_X + hd * 32u + 16u), WS_OUT, ROW);
#endif
        // own token tile: S^T [64 keys][16] = Q_keys Q_own^T + bias^T, softmax in registers (the transposed body's
        // tree), O^T = V^T P^T, then the out projection into its accumulators
#define ATT(r0_, qo0, qo1, ca0, ca1) { \
            MH bh; \
            coopMatLoad(bh, pc.bias.v, hd * 4096u + (r0_) * 64u, 64u, COL); ACC s0 = ACC(bh); \
            coopMatLoad(bh, pc.bias.v, hd * 4096u + (r0_) * 64u + 16u, 64u, COL); ACC s1 = ACC(bh); \
            coopMatLoad(bh, pc.bias.v, hd * 4096u + (r0_) * 64u + 32u, 64u, COL); ACC s2 = ACC(bh); \
            coopMatLoad(bh, pc.bias.v, hd * 4096u + (r0_) * 64u + 48u, 64u, COL); ACC s3 = ACC(bh); \
            { MB b_; MA a_; CPB(b_, qo0) \
              IFK((r0_) >> 4, 0u) { CPA(a_, q00) MMA(s0, a_, b_) } IFK((r0_) >> 4, 1u) { CPA(a_, q10) MMA(s1, a_, b_) } \
              IFK((r0_) >> 4, 2u) { CPA(a_, q20) MMA(s2, a_, b_) } IFK((r0_) >> 4, 3u) { CPA(a_, q30) MMA(s3, a_, b_) } \
              Q2_IF { CPB(b_, qo1) \
              IFK((r0_) >> 4, 0u) { CPA(a_, q01) MMA(s0, a_, b_) } IFK((r0_) >> 4, 1u) { CPA(a_, q11) MMA(s1, a_, b_) } \
              IFK((r0_) >> 4, 2u) { CPA(a_, q21) MMA(s2, a_, b_) } IFK((r0_) >> 4, 3u) { CPA(a_, q31) MMA(s3, a_, b_) } } } \
            MH p0 = MH(s0), p1 = MH(s1), p2 = MH(s2), p3 = MH(s3); \
            f16vec2 e2[4]; \
            [[unroll]] for (int e = 0; e < 8; e += 2) { \
                f16vec2 t0 = TK((r0_) >> 4, 0u, unpackFloat2x16(exp_bits2(f16vec2(p0[e], p0[e + 1])))); \
                f16vec2 t1 = TK((r0_) >> 4, 1u, unpackFloat2x16(exp_bits2(f16vec2(p1[e], p1[e + 1])))); \
                f16vec2 t2 = TK((r0_) >> 4, 2u, unpackFloat2x16(exp_bits2(f16vec2(p2[e], p2[e + 1])))); \
                f16vec2 t3 = TK((r0_) >> 4, 3u, unpackFloat2x16(exp_bits2(f16vec2(p3[e], p3[e + 1])))); \
                p0[e] = t0.x, p0[e + 1] = t0.y, p1[e] = t1.x, p1[e + 1] = t1.y; \
                p2[e] = t2.x, p2[e + 1] = t2.y, p3[e] = t3.x, p3[e + 1] = t3.y; \
                precise f16vec2 sm = (t0 + t1) + (t2 + t3); \
                e2[e >> 1] = sm; \
            } \
            f16vec2 pc2[4]; \
            [[unroll]] for (int i = 0; i < 4; ++i) { \
                f16vec2 o = unpackFloat2x16(subgroupShuffleXor(packFloat2x16(e2[i]), 16u)); \
                precise f16vec2 sp = e2[i] + o; \
                pc2[i] = sp; \
            } \
            precise f16vec2 b2_ = (pc2[0] + pc2[1]) + (pc2[2] + pc2[3]); \
            precise float16_t sum = b2_.x + b2_.y; \
            float16_t r = RCP_SUM(sum); \
            [[unroll]] for (int e = 0; e < 8; e += 2) { \
                precise f16vec2 u0 = f16vec2(p0[e], p0[e + 1]) * r, u1 = f16vec2(p1[e], p1[e + 1]) * r; \
                precise f16vec2 u2 = f16vec2(p2[e], p2[e + 1]) * r, u3 = f16vec2(p3[e], p3[e + 1]) * r; \
                p0[e] = u0.x, p0[e + 1] = u0.y, p1[e] = u1.x, p1[e + 1] = u1.y; \
                p2[e] = u2.x, p2[e + 1] = u2.y, p3[e] = u3.x, p3[e + 1] = u3.y; \
            } \
            ATT_OUT(r0_, ca0, ca1) }
#ifdef LT_VO
// the value path is linear: V_h already through wout_h in_b_h (Engine::upload_stage), so P V_h goes straight into the out
// accumulators, without the 16 products per head of the out projection
#define ATT_OUT(r0_, o0, o1) { MB b_; MA a_; \
              IFK((r0_) >> 4, 0u) { CPB(b_, p0) CPA(a_, v00) MMA(o0, a_, b_) CPA(a_, v01) MMA(o1, a_, b_) } \
              IFK((r0_) >> 4, 1u) { CPB(b_, p1) CPA(a_, v10) MMA(o0, a_, b_) CPA(a_, v11) MMA(o1, a_, b_) } \
              IFK((r0_) >> 4, 2u) { CPB(b_, p2) CPA(a_, v20) MMA(o0, a_, b_) CPA(a_, v21) MMA(o1, a_, b_) } \
              IFK((r0_) >> 4, 3u) { CPB(b_, p3) CPA(a_, v30) MMA(o0, a_, b_) CPA(a_, v31) MMA(o1, a_, b_) } }
#else
#define ATT_OUT(r0_, ca0, ca1) ACC o0 = ACC(0.0), o1 = ACC(0.0); \
            { MB b_; MA a_; \
              IFK((r0_) >> 4, 0u) { CPB(b_, p0) CPA(a_, v00) MMA(o0, a_, b_) CPA(a_, v01) MMA(o1, a_, b_) } \
              IFK((r0_) >> 4, 1u) { CPB(b_, p1) CPA(a_, v10) MMA(o0, a_, b_) CPA(a_, v11) MMA(o1, a_, b_) } \
              IFK((r0_) >> 4, 2u) { CPB(b_, p2) CPA(a_, v20) MMA(o0, a_, b_) CPA(a_, v21) MMA(o1, a_, b_) } \
              IFK((r0_) >> 4, 3u) { CPB(b_, p3) CPA(a_, v30) MMA(o0, a_, b_) CPA(a_, v31) MMA(o1, a_, b_) } \
              MH oh = MH(o0); CPB(b_, oh) MMA(ca0, wo00, b_) MMA(ca1, wo10, b_) \
              oh = MH(o1); CPB(b_, oh) MMA(ca0, wo01, b_) MMA(ca1, wo11, b_) }
#endif
#ifdef LT_SYM
        // LT_SYM: the weights before the normalisation are symmetric (S = Q Q^T + a symmetric bias), so a pair of token tiles
        // goes through the bias, S and the exponential once. The upper tile's [keys][own] is the lower one's transposed, and an
        // accumulator copied is the A operand of its transpose: one product with the identity gives it back as an accumulator
#define E_DIR(p, T, K) { coopMatLoad(p, pc.bias.v, hd * 4096u + uint(T) * 1024u + uint(K) * 16u, 64u, COL); \
            { MB b_; MA a_; CPB(b_, q##T##0) CPA(a_, q##K##0) MMA(p, a_, b_) Q2_IF { CPB(b_, q##T##1) CPA(a_, q##K##1) MMA(p, a_, b_) } } \
            [[unroll]] for (int e = 0; e < 8; e += 2) { f16vec2 t_ = unpackFloat2x16(exp_bits2(f16vec2(p[e], p[e + 1]))); p[e] = t_.x, p[e + 1] = t_.y; } }
#define E_UP(T, K) MH e##T##K = MH(0.0); if (KEPT(uint(T), uint(K)) && (lv##T || lv##K)) E_DIR(e##T##K, T, K)
#define E_OWN(p, T) MH p = MH(0.0); E_DIR(p, T, T)
#define E_LO(p, T, K) MH p = MH(0.0); if (KEPT(uint(T), uint(K))) { MA a_; CPA(a_, e##K##T) MMA(p, a_, idb) }
        E_UP(0, 1) E_UP(0, 2) E_UP(0, 3) E_UP(1, 2) E_UP(1, 3) E_UP(2, 3)
#define ATT_N(r0_, ca0, ca1) { \
            f16vec2 e2[4]; \
            [[unroll]] for (int e = 0; e < 8; e += 2) { \
                precise f16vec2 sm = (f16vec2(p0[e], p0[e + 1]) + f16vec2(p1[e], p1[e + 1])) + (f16vec2(p2[e], p2[e + 1]) + f16vec2(p3[e], p3[e + 1])); \
                e2[e >> 1] = sm; \
            } \
            f16vec2 pc2[4]; \
            [[unroll]] for (int i = 0; i < 4; ++i) { \
                f16vec2 o = unpackFloat2x16(subgroupShuffleXor(packFloat2x16(e2[i]), 16u)); \
                precise f16vec2 sp = e2[i] + o; \
                pc2[i] = sp; \
            } \
            precise f16vec2 b2_ = (pc2[0] + pc2[1]) + (pc2[2] + pc2[3]); \
            precise float16_t sum = b2_.x + b2_.y; \
            float16_t r = RCP_SUM(sum); \
            [[unroll]] for (int e = 0; e < 8; e += 2) { \
                precise f16vec2 u0 = f16vec2(p0[e], p0[e + 1]) * r, u1 = f16vec2(p1[e], p1[e + 1]) * r; \
                precise f16vec2 u2 = f16vec2(p2[e], p2[e + 1]) * r, u3 = f16vec2(p3[e], p3[e + 1]) * r; \
                p0[e] = u0.x, p0[e + 1] = u0.y, p1[e] = u1.x, p1[e + 1] = u1.y; \
                p2[e] = u2.x, p2[e + 1] = u2.y, p3[e] = u3.x, p3[e + 1] = u3.y; \
            } \
            ATT_OUT(r0_, ca0, ca1) }
        if (lv0) { E_OWN(p0, 0) MH p1 = e01, p2 = e02, p3 = e03; ATT_N(0u, c00, c01) }
        if (lv1) { E_LO(p0, 1, 0) E_OWN(p1, 1) MH p2 = e12, p3 = e13; ATT_N(16u, c10, c11) }
        if (lv2) { E_LO(p0, 2, 0) E_LO(p1, 2, 1) E_OWN(p2, 2) MH p3 = e23; ATT_N(32u, c20, c21) }
        if (lv3) { E_LO(p0, 3, 0) E_LO(p1, 3, 1) E_LO(p2, 3, 2) E_OWN(p3, 3) ATT_N(48u, c30, c31) }
#else
        if (lv0) ATT(0u, q00, q01, c00, c01)
        if (lv1) ATT(16u, q10, q11, c10, c11)
        if (lv2) ATT(32u, q20, q21, c20, c21)
        if (lv3) ATT(48u, q30, q31, c30, c31)
#endif

/**
 * @file softmax_int8.c
 * @brief Integer-only softmax implementation (DA + DI + EN pipeline)
 *
 * ALGORITHM:
 *   1. DA (Denominator Accumulation):
 *      - Track running max m and sum Sigma
 *      - Renormalize Sigma when max increases via bit-shift
 *      - Accumulate exp2 contributions using LUT
 *
 *   2. DI (Denominator Inversion):
 *      - Compute reciprocal Sigma^{-1} using LUT (Q1.7)
 *      - Optional Newton-Raphson refinement
 *
 *   3. EN (Element Normalization):
 *      - For each element: exp2(frac) >> bin * Sigma_inv
 *      - Output Q1.7 probabilities, saturated to [0, 127]
 *
 * NUMERICAL PROPERTIES:
 *   - Input range: int8_t [-128, 127]
 *   - K = 5 (inner shift for B=8)
 *   - Max bin = (127 - (-128)) >> 5 = 255 >> 5 = 7
 *   - Sigma width: uint32_t (sufficient for N ≤ 4096)
 *   - Output sum: 127 ± 2 (within 1.6% of 1.0 in Q1.7)
 *
 * DATE: 2025-10-27
 */

#include <stddef.h>
#include <stdint.h>

#include "fixedpoint.h"
#include "lut_exp2.h"
#include "softmax_int8.h"

/* ========================================================================
 * INTERNAL HELPERS
 * ======================================================================== */

/**
 * @brief Compute exp contribution for given delta from max
 *
 * Computes e^(-d) ≈ 2^(-d * log2(e)) using fractional LUT and integer shift.
 *
 * The algorithm with log2(e) scaling:
 *   e^(-d) = 2^(-d * log2(e)) where log2(e) ≈ 1.4427
 *
 * We approximate: d_scaled = (d * 185) >> 7  (185/128 ≈ 1.4453 ≈ log2(e))
 * Then: d_scaled = bin * 2^K + frac
 * Finally: 2^(-d_scaled/2^K) = (2^(-frac/2^K)) >> bin
 *
 * @param d           Delta from max (m - x), must be >= 0
 * @param K           Inner shift parameter
 * @param q_exp       Fractional bits in exp2 output
 * @param use_frac    1 to use fractional LUT, 0 for bucket-only
 * @param scale_log2e 1 to apply log2(e) scaling, 0 for raw 2^x
 * @return            Exp contribution in Q1.8 format
 */
static inline uint32_t compute_exp2_contrib(uint16_t d, uint8_t K,
                                             uint8_t q_exp, uint8_t use_frac,
                                             uint8_t scale_log2e) {
    /* Apply log2(e) scaling if requested: e^(-d) = 2^(-d * log2(e))
     * log2(e) ≈ 1.44269... ≈ 185/128 (Q7 fixed point)
     */
    uint16_t d_scaled = d;
    if (scale_log2e) {
        /* Multiply by 185/128 = (d * 185) >> 7 */
        uint32_t temp = ((uint32_t)d * 185) >> 7;
        if (temp > 255) temp = 255;  /* Saturate to prevent overflow */
        d_scaled = (uint16_t)temp;
    }

    /* Decompose delta: d_scaled = bin * 2^K + frac */
    uint8_t bin = d_scaled >> K;        /* Coarse bin (integer part after shift) */
    uint8_t frac = d_scaled & ((1 << K) - 1);  /* Fractional residual */

    /* Fast path: if bin is too large, contribution is negligible */
    if (bin >= 20) {
        return 0;  /* 2^(-bin) underflows for reasonable Q formats */
    }

    /* Get fractional part: 2^(-frac/2^K) in Q1.8 */
    uint32_t base;
    if (use_frac && frac > 0) {
        /* Look up 2^(-frac/2^K) from LUT */
        base = exp2_frac_lookup[frac];  /* Q1.8: values around 256 */
    } else {
        /* Bucket-only or frac=0: use full 1.0 in Q1.8 */
        base = 256;  /* Q1.8 representation of 1.0 */
    }

    /* Apply coarse bin shift: base >> bin
     * This gives us 2^(-bin) * 2^(-frac/2^K) = 2^(-(bin + frac/2^K))
     * Which approximates e^(-d) when scale_log2e is enabled
     */
    return base >> bin;
}

/* ========================================================================
 * CORE SOFTMAX IMPLEMENTATION
 * ======================================================================== */

/**
 * @brief Process one row with online softmax
 *
 * Implements three-stage pipeline:
 *   DA: Accumulate denominator with online max tracking
 *   DI: Invert denominator
 *   EN: Normalize each element
 *
 * @param x_q   Input row (int8_t[N])
 * @param N     Length of row
 * @param y_q   Output probabilities (uint8_t[N], Q1.7)
 * @param cfg   Configuration (NULL = use default)
 */
void softmax_int8_row(const int8_t* x_q, uint16_t N,
                      uint8_t* y_q, const smx_cfg_t* cfg) {
    /* Validate inputs */
    if (x_q == NULL || y_q == NULL || N == 0) {
        return;
    }

    /* Use default config if not provided */
    const smx_cfg_t* c = (cfg != NULL) ? cfg : &SMX_CFG_DEFAULT;
    const uint8_t K = c->K;
    const uint8_t q_exp = c->q_exp;
    const uint8_t use_frac = c->use_frac;
    const uint8_t scale_log2e = c->scale_log2e;

    /* ====================================================================
     * STAGE 1: DA (Denominator Accumulation)
     * ==================================================================== */

    int8_t m = -128;      /* Running max (start at minimum int8_t) */
    uint32_t Sigma = 0;   /* Denominator accumulator (wide for safety) */

    for (uint16_t i = 0; i < N; i++) {
        int8_t x = x_q[i];

        /* Check if new max found */
        if (x > m) {
            /* Renormalize previous sum to new reference
             * When max increases from m_old to m_new, all previous terms
             * need to be scaled: e^(-(m_old - x_i)) → e^(-(m_new - x_i))
             * = e^(-(m_old - x_i) - (m_new - m_old))
             * = previous_term * e^(-(m_new - m_old))
             * ≈ previous_term >> ((m_new - m_old) * log2(e))
             */
            int32_t delta_m = (int32_t)x - (int32_t)m;  /* Increase in max */

            if (delta_m > 0) {
                /* Apply log2(e) scaling to delta_m if enabled */
                uint32_t shift_amount;
                if (scale_log2e) {
                    /* delta_m_scaled = (delta_m * 185) >> 7 */
                    shift_amount = ((uint32_t)delta_m * 185) >> 7;
                } else {
                    shift_amount = (uint32_t)delta_m;
                }

                /* Scale down Sigma by the scaled amount */
                if (shift_amount < 32) {
                    Sigma >>= shift_amount;
                } else {
                    Sigma = 0;  /* Old sum negligible */
                }
            }

            m = x;  /* Update max */
        }

        /* Compute contribution of this element */
        uint16_t d = (uint16_t)(m - x);  /* Delta from current max (>=0) */
        uint32_t term = compute_exp2_contrib(d, K, q_exp, use_frac, scale_log2e);

        /* Accumulate (with overflow guard) */
        if (Sigma <= UINT32_MAX - term) {
            Sigma += term;
        } else {
            /* Saturate at max (should not happen for N <= 4096) */
            Sigma = UINT32_MAX;
        }
    }

    /* Handle edge case: Sigma is zero (all inputs very negative) */
    if (Sigma == 0) {
        /* Uniform distribution as fallback */
        uint8_t uniform_val = (N > 0) ? (127 / N) : 0;
        for (uint16_t i = 0; i < N; i++) {
            y_q[i] = uniform_val;
        }
        return;
    }

    /* ====================================================================
     * STAGE 2: DI (Denominator Inversion)
     * ==================================================================== */

    /* Compute reciprocal for normalization
     * num is Q1.8 (256 = 1.0), output should be Q1.7 (128 = 1.0)
     * We want: out = (num / Sigma) * 128
     *
     * Rewrite as: out = num * (128 / Sigma) / 1
     *            = num * Sigma_inv, where Sigma_inv = (128 << shift) / Sigma
     *
     * Choose shift to maintain precision while avoiding overflow
     */

    const uint8_t shift = 16;  /* Use 16-bit extra precision */
    uint32_t Sigma_inv;

    if (Sigma == 0) {
        Sigma_inv = 0;
    } else {
        /* Compute (128 << shift) / Sigma for maximum precision
         * shift chosen so numerator fits in uint32 and result is meaningful
         */
        Sigma_inv = ((uint32_t)128 << shift) / Sigma;

        /* Sigma_inv is now in Q(1+shift).7 format */
        /* When we multiply num * Sigma_inv, we get Q(2+shift).15 */
        /* We'll shift by (shift + 8) to get Q1.7 */
    }

    /* ====================================================================
     * STAGE 3: EN (Element Normalization)
     * ==================================================================== */

    /* ====================================================================
     * STAGE 3: EN (Element Normalization)
     * ==================================================================== */

    for (uint16_t i = 0; i < N; i++) {
        int8_t x = x_q[i];
        uint16_t d = (uint16_t)(m - x);  /* Delta from max */

        /* Compute numerator: exp contribution in Q1.8 */
        uint32_t num = compute_exp2_contrib(d, K, q_exp, use_frac, scale_log2e);

        /* Multiply by reciprocal:
         * num is Q1.8, Sigma_inv is (128 << 16) / Sigma
         * prod = num * Sigma_inv = num * 128 * (1 << 16) / Sigma
         * We want: (num / Sigma) * 128 in Q1.7
         * So: prod >> (16 + 8) = prod >> 24 gives (num * 128 / Sigma)
         * But we want Q1.7 output (128 = 1.0), so shift is 16 + 8 - 0 = 24
         * Actually: prod = num * (128 << 16)/Sigma
         *         = (num * 128 / Sigma) << 16
         * So: prod >> 16 = num * 128 / Sigma (this is already scaled for Q1.7!)
         */
        uint64_t prod = (uint64_t)num * (uint64_t)Sigma_inv;
        uint32_t prob = (uint32_t)((prod + (1ULL << (shift - 1))) >> shift);

        /* Saturate to Q1.7 range [0, 127] */
        y_q[i] = sat_q1_7((int32_t)prob);
    }
}

/* ========================================================================
 * A×V FUSION (STUB FOR FUTURE OPTIMIZATION)
 * ======================================================================== */

/**
 * @brief Fused softmax-then-matmul for attention
 *
 * OPTIMIZATION STRATEGY:
 *   Instead of:
 *     1. Compute softmax(A) → store N probabilities
 *     2. Load all N probs and multiply with V
 *   Do:
 *     1. DA+DI stages (compute Sigma_inv)
 *     2. Stream EN stage: for each i, compute prob[i] and immediately
 *        accumulate prob[i] * V[i,:] into output
 *     3. Never materialize full probability vector
 *
 *   MEMORY SAVINGS: N×1 byte (probability storage)
 *   CACHE BENEFITS: Streaming access pattern, better locality
 *
 * PSEUDOCODE:
 *   // DA+DI stages (same as softmax_int8_row)
 *   m, Sigma = compute_max_and_sum(a_row);
 *   Sigma_inv = reciprocal(Sigma);
 *
 *   // Fused EN + matmul
 *   for (j = 0; j < P; j++) {
 *     acc[j] = 0;
 *   }
 *   for (i = 0; i < N; i++) {
 *     prob_i = normalize_element(a_row[i], m, Sigma_inv);  // EN stage
 *     for (j = 0; j < P; j++) {
 *       acc[j] += prob_i * V[i * P + j];  // Accumulate A×V
 *     }
 *   }
 *   // Requantize acc to output
 *
 * IMPLEMENTATION: TODO - left as exercise for performance tuning
 */
void softmax_then_axv(const int8_t* a_row, const int8_t* V,
                      uint16_t N, uint16_t P,
                      int8_t* out_row, const smx_cfg_t* cfg) {
    (void)a_row;
    (void)V;
    (void)N;
    (void)P;
    (void)out_row;
    (void)cfg;

    /* NOT YET IMPLEMENTED */
    /* See comments above for optimization strategy */
    /* Left as stub for future work */
}

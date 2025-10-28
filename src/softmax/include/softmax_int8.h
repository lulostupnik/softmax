/**
 * @file softmax_int8.h
 * @brief Integer-only Softmax for INT8 attention scores
 *
 * Hardware-friendly online softmax using base-2 exponent with LUT-based
 * fractional approximation. Zero floating-point, shift-based scaling only.
 *
 * FORMATS:
 *   - Inputs:  int8_t x_q[N]      (B=8 signed integers)
 *   - Outputs: uint8_t y_q[N]     (Q1.7 probabilities: 0..127 ~ 0.0..~1.0)
 *   - Internal Sigma: uint32_t    (wide accumulator)
 *   - Sigma_inv: uint16_t         (Q1.7 reciprocal)
 *
 * ALGORITHM:
 *   DA: Track running max m and denominator Sigma with renormalization
 *   DI: Compute reciprocal Sigma^{-1} via LUT (optional Newton step)
 *   EN: Normalize each element: exp2_lut[frac] >> bin * Sigma_inv
 *
 * PARAMETERS:
 *   K = 5 (inner right-shift for B=8)
 *   Qexp = 8 (fractional bits in exp2 LUT output)
 *   Qprob = 7 (fractional bits in output probabilities)
 *
 * CONSTRAINTS:
 *   - Portable C99, reentrant, no dynamic allocation
 *   - N ≤ 4096 typical
 *   - Sum of outputs ≈ 1.0 in Q1.7 (127 ± 2)
 *
 * DATE: 2025-10-27
 */

#ifndef SOFTMAX_INT8_H
#define SOFTMAX_INT8_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Configuration for softmax computation
 */
typedef struct
{
    uint8_t K;        /**< Inner right-shift (default 5 for B=8) */
    uint8_t q_exp;    /**< Fractional bits of exp2() output (default 8) */
    uint8_t q_prob;   /**< Fractional bits of output probs (default 7) */
    uint8_t use_frac; /**< 0: bucket only, 1: use fractional LUT/PWL */
    uint8_t scale_log2e; /**< 1: apply log2(e) scaling for e^x approx, 0: raw 2^x */
} smx_cfg_t;

/**
 * @brief Default configuration for B=8 inputs (raw 2^x, no e scaling)
 */
static const smx_cfg_t SMX_CFG_DEFAULT = {
    .K = 5,  /* K = B - log2(B) = 8 - 3 = 5 */
    .q_exp = 8,
    .q_prob = 7,
    .use_frac = 1,
    .scale_log2e = 0  /* Disable for now - raw 2^x */
};

/**
 * @brief Bucket-only configuration (fast, less accurate)
 */
static const smx_cfg_t SMX_CFG_BUCKET_ONLY = {
    .K = 5,  /* K = B - log2(B) = 8 - 3 = 5 */
    .q_exp = 8,
    .q_prob = 7,
    .use_frac = 0,
    .scale_log2e = 0  /* Disable for now */
};

/**
 * @brief Process one row with online softmax (DA + DI + EN pipeline)
 *
 * Computes softmax over int8 row using integer-only operations:
 *   y[i] = exp2(-(m - x[i])/2^K) / sum_j exp2(-(m - x[j])/2^K)
 *
 * @param x_q   Input row (int8_t[N])
 * @param N     Length of row (must be > 0, ≤ 4096 typical)
 * @param y_q   Output probabilities (uint8_t[N], Q1.7 format)
 * @param cfg   Configuration (NULL uses SMX_CFG_DEFAULT)
 *
 * @note Output y_q can alias x_q if needed (processed in two passes)
 * @note Thread-safe and reentrant
 */
void softmax_int8_row(const int8_t* x_q, uint16_t N,
                      uint8_t* y_q, const smx_cfg_t* cfg);

/**
 * @brief Normalize-and-matmul in one pass (A×V fusion)
 *
 * Fuses softmax normalization with attention×value multiplication:
 *   out[j] = sum_i softmax(a_row[i]) * V[i,j]
 *
 * Reduces memory traffic by streaming normalize-and-accumulate.
 *
 * @param a_row     Attention row to normalize (int8_t[N])
 * @param V         Value matrix (int8_t[N×P], row-major)
 * @param N         Attention dimension
 * @param P         Value dimension
 * @param out_row   Output row (int8_t[P])
 * @param cfg       Configuration (NULL uses SMX_CFG_DEFAULT)
 *
 * @note NOT YET IMPLEMENTED - stub for future optimization
 */
void softmax_then_axv(const int8_t* a_row, const int8_t* V,
                      uint16_t N, uint16_t P,
                      int8_t* out_row, const smx_cfg_t* cfg);

#ifdef __cplusplus
}
#endif

#endif /* SOFTMAX_INT8_H */

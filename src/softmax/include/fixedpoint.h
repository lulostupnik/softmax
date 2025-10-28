/**
 * @file fixedpoint.h
 * @brief Fixed-point Q-format helpers for integer-only computation
 *
 * Provides saturating arithmetic, Q-format multiplies with rounding,
 * and conversion utilities for signed/unsigned fixed-point values.
 *
 * Q-FORMAT NOTATION:
 *   Qm.n = m integer bits, n fractional bits
 *   Q1.7 = 1 sign + 1 integer + 7 fractional = 8 bits total
 *   Value range: [-2.0, +1.9921875] for signed, [0, 1.9921875] for unsigned
 *
 * DATE: 2025-10-27
 */

#ifndef FIXEDPOINT_H
#define FIXEDPOINT_H

#include <limits.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Q-FORMAT CONSTANTS
 * ======================================================================== */

#define Q1_7_SCALE   (1 << 7)   /**< 2^7 = 128 */
#define Q1_8_SCALE   (1 << 8)   /**< 2^8 = 256 */
#define Q1_7_MAX     127        /**< Max value in unsigned Q1.7 (~1.0) */
#define Q1_8_MAX     255        /**< Max value in unsigned Q1.8 (~1.0) */

/* ========================================================================
 * SATURATION HELPERS
 * ======================================================================== */

/**
 * @brief Saturate to uint8_t range [0, 255]
 */
static inline uint8_t sat_u8(int32_t x) {
    if (x < 0) return 0;
    if (x > 255) return 255;
    return (uint8_t)x;
}

/**
 * @brief Saturate to int8_t range [-128, 127]
 */
static inline int8_t sat_i8(int32_t x) {
    if (x < -128) return -128;
    if (x > 127) return 127;
    return (int8_t)x;
}

/**
 * @brief Saturate to uint16_t range [0, 65535]
 */
static inline uint16_t sat_u16(int32_t x) {
    if (x < 0) return 0;
    if (x > 65535) return 65535;
    return (uint16_t)x;
}

/**
 * @brief Saturate to Q1.7 range [0, 127]
 */
static inline uint8_t sat_q1_7(int32_t x) {
    if (x < 0) return 0;
    if (x > Q1_7_MAX) return Q1_7_MAX;
    return (uint8_t)x;
}

/* ========================================================================
 * Q-FORMAT MULTIPLY WITH ROUNDING
 * ======================================================================== */

/**
 * @brief Multiply two Q-format numbers with rounding
 *
 * Computes: (a * b) >> frac_bits with rounding to nearest
 *
 * @param a          First operand
 * @param b          Second operand
 * @param frac_bits  Number of fractional bits to shift out
 * @return           Rounded product
 *
 * @note Uses rounding: add (1 << (frac_bits-1)) before shift
 */
static inline int32_t mul_q(int32_t a, int32_t b, uint8_t frac_bits) {
    int64_t prod = (int64_t)a * (int64_t)b;
    if (frac_bits == 0) return (int32_t)prod;

    /* Round to nearest: add 0.5 in Q-format before truncation */
    int64_t round_bias = (int64_t)1 << (frac_bits - 1);
    return (int32_t)((prod + round_bias) >> frac_bits);
}

/**
 * @brief Multiply unsigned Q-format numbers (e.g., Q1.7 × Q1.8 → Q1.7)
 *
 * Typical usage: multiply exp2 output (Q1.8) by reciprocal (Q1.7)
 *
 * @param a          First operand (e.g., Q1.8 numerator)
 * @param b          Second operand (e.g., Q1.7 reciprocal)
 * @param frac_bits  Total fractional bits to remove
 * @return           Saturated uint8 result
 */
static inline uint8_t mul_q_sat_u8(uint16_t a, uint16_t b, uint8_t frac_bits) {
    uint32_t prod = (uint32_t)a * (uint32_t)b;
    if (frac_bits == 0) return sat_u8((int32_t)prod);

    uint32_t round_bias = (uint32_t)1 << (frac_bits - 1);
    uint32_t result = (prod + round_bias) >> frac_bits;
    return sat_u8((int32_t)result);
}

/* ========================================================================
 * CONVERSION UTILITIES
 * ======================================================================== */

/**
 * @brief Convert float to Q1.7 (for testing only)
 */
static inline uint8_t float_to_q1_7(float x) {
    int32_t val = (int32_t)(x * Q1_7_SCALE + 0.5f);
    return sat_q1_7(val);
}

/**
 * @brief Convert Q1.7 to float (for testing only)
 */
static inline float q1_7_to_float(uint8_t x) {
    return (float)x / (float)Q1_7_SCALE;
}

/**
 * @brief Convert float to Q1.8 (for testing only)
 */
static inline uint8_t float_to_q1_8(float x) {
    int32_t val = (int32_t)(x * Q1_8_SCALE + 0.5f);
    return sat_u8(val);
}

/**
 * @brief Convert Q1.8 to float (for testing only)
 */
static inline float q1_8_to_float(uint8_t x) {
    return (float)x / (float)Q1_8_SCALE;
}

#ifdef __cplusplus
}
#endif

#endif /* FIXEDPOINT_H */

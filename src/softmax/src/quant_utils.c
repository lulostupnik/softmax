/**
 * @file quant_utils.c
 * @brief Quantization and fixed-point utilities
 *
 * Additional Q-format helpers beyond what's in fixedpoint.h header.
 * Provides runtime utilities for clipping, scaling, and debugging.
 *
 * DATE: 2025-10-27
 */

#include <limits.h>
#include <stdint.h>

#include "fixedpoint.h"

/* ========================================================================
 * CLIPPING AND RANGE UTILITIES
 * ======================================================================== */

/**
 * @brief Clip value to specified range
 *
 * @param x    Input value
 * @param min  Minimum allowed value
 * @param max  Maximum allowed value
 * @return     Clipped value
 */
int32_t clip_i32(int32_t x, int32_t min, int32_t max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

/**
 * @brief Safe right shift with rounding
 *
 * Performs arithmetic right shift with round-to-nearest.
 * Handles shift >= 32 gracefully.
 *
 * @param x      Value to shift
 * @param shift  Number of bits to shift (0-31 valid)
 * @return       Rounded shifted value
 */
int32_t rshift_round(int32_t x, uint8_t shift) {
    if (shift == 0) return x;
    if (shift >= 31) return (x < 0) ? -1 : 0;

    int32_t round_bias = (int32_t)1 << (shift - 1);
    return (x + round_bias) >> shift;
}

/**
 * @brief Safe unsigned right shift with rounding
 *
 * @param x      Value to shift
 * @param shift  Number of bits to shift
 * @return       Rounded shifted value
 */
uint32_t urshift_round(uint32_t x, uint8_t shift) {
    if (shift == 0) return x;
    if (shift >= 32) return 0;

    uint32_t round_bias = (uint32_t)1 << (shift - 1);
    return (x + round_bias) >> shift;
}

/* ========================================================================
 * OVERFLOW DETECTION
 * ======================================================================== */

/**
 * @brief Check if addition would overflow int32_t
 *
 * @param a  First operand
 * @param b  Second operand
 * @return   1 if overflow would occur, 0 otherwise
 */
int would_overflow_i32_add(int32_t a, int32_t b) {
    if (b > 0 && a > INT32_MAX - b) return 1;
    if (b < 0 && a < INT32_MIN - b) return 1;
    return 0;
}

/**
 * @brief Check if multiplication would overflow int32_t
 *
 * @param a  First operand
 * @param b  Second operand
 * @return   1 if overflow would occur, 0 otherwise
 */
int would_overflow_i32_mul(int32_t a, int32_t b) {
    if (a == 0 || b == 0) return 0;

    /* Use 64-bit intermediate to check */
    int64_t result = (int64_t)a * (int64_t)b;
    return (result > INT32_MAX || result < INT32_MIN);
}

/* ========================================================================
 * DEBUGGING AND VALIDATION HELPERS
 * ======================================================================== */

/**
 * @brief Count leading zeros (8-bit)
 *
 * Useful for dynamic range detection and normalization.
 *
 * @param x  Input value
 * @return   Number of leading zero bits
 */
uint8_t clz_u8(uint8_t x) {
    if (x == 0) return 8;

    uint8_t n = 0;
    if (x <= 0x0F) { n += 4; x <<= 4; }
    if (x <= 0x3F) { n += 2; x <<= 2; }
    if (x <= 0x7F) { n += 1; }

    return n;
}

/**
 * @brief Count leading zeros (16-bit)
 *
 * @param x  Input value
 * @return   Number of leading zero bits
 */
uint8_t clz_u16(uint16_t x) {
    if (x == 0) return 16;

    uint8_t n = 0;
    if (x <= 0x00FF) { n += 8; x <<= 8; }
    if (x <= 0x0FFF) { n += 4; x <<= 4; }
    if (x <= 0x3FFF) { n += 2; x <<= 2; }
    if (x <= 0x7FFF) { n += 1; }

    return n;
}

/**
 * @brief Find minimum shift to fit value in 8 bits
 *
 * Used for dynamic scaling of accumulators.
 *
 * @param x  Input value (uint32_t)
 * @return   Minimum right shift needed to fit in uint8_t
 */
uint8_t min_shift_to_u8(uint32_t x) {
    if (x <= 0xFF) return 0;

    uint8_t shift = 0;
    while (x > 0xFF) {
        x >>= 1;
        shift++;
    }

    return shift;
}

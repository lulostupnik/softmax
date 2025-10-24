#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Integer-only softmax.
 * - Input:  int8_t x[i]
 * - Output: int8_t y[i] in [0, 127], approximately summing to 127
 * - Internals: Q15 fixed-point for exp(-d) via a tiny LUT (no floats used)
 *
 * Notes:
 *  - We subtract the max input for numerical stability.
 *  - exp(-d) is negligible in Q15 for d > 10, so we clamp at 0 beyond that.
 */

// Q15 LUT for exp(-d), d = 0..10  (round(exp(-d)*32768))
// Beyond d=10 the value is ~0 in Q15.
static const uint16_t EXP_NEG_Q15_LUT[11] = {
    32768, // exp(0)   * 32768
    12055, // exp(-1)  * 32768
    4435,  // exp(-2)  * 32768
    1631,  // exp(-3)  * 32768
    600,   // exp(-4)  * 32768
    221,   // exp(-5)  * 32768
    81,    // exp(-6)  * 32768
    30,    // exp(-7)  * 32768
    11,    // exp(-8)  * 32768
    4,     // exp(-9)  * 32768
    1      // exp(-10) * 32768
};

// Integer-only softmax with int8 input and int8 output (0..127).
void softmax_int8_pureint(const int8_t *input, int8_t *output, int length) {
    if (length <= 0) return;

    // 1) Find max value for numerical stability
    int8_t max_val = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    // 2) Compute "weights" = exp(input[i] - max_val) using LUT in Q15
    //    and accumulate sum (use 32-bit to avoid overflow).
    uint32_t sum_w = 0;
    uint16_t *w = (uint16_t*)alloca(sizeof(uint16_t) * (size_t)length);

    for (int i = 0; i < length; i++) {
        // delta = max - x (non-negative)
        int d = (int)max_val - (int)input[i];  // d in [0..255] theoretically
        uint16_t wi;
        if (d <= 10) {
            wi = EXP_NEG_Q15_LUT[d];
        } else {
            wi = 0; // negligible in Q15 for d > 10
        }
        w[i] = wi;
        sum_w += wi;
    }

    // Edge case: if everything underflowed to zero (e.g., all d>10)
    // set argmax to 127 and the rest to 0.
    if (sum_w == 0) {
        // Find first argmax index (any max is fine)
        int imax = 0;
        for (int i = 1; i < length; i++) {
            if (input[i] > input[imax]) imax = i;
        }
        for (int i = 0; i < length; i++) output[i] = 0;
        output[imax] = 127;
        return;
    }

    // 3) Normalize to int8 in [0..127]:
    //    y_i = round(127 * w_i / sum_w)
    for (int i = 0; i < length; i++) {
        // Multiply first in 32-bit, add half divisor for rounding
        uint32_t num = (uint32_t)w[i] * 127u;
        int scaled = (int)((num + (sum_w >> 1)) / sum_w);
        if (scaled > 127) scaled = 127;
        if (scaled < 0)   scaled = 0;
        output[i] = (int8_t)scaled;
    }
}

int main(void) {
    int8_t input[] = {1, 2, 3, 4};
    int n = (int)(sizeof(input) / sizeof(input[0]));
    int8_t out[n];

    softmax_int8_pureint(input, out, n);

    printf("Input:  ");
    for (int i = 0; i < n; i++) printf("%d ", input[i]);
    printf("\nSoftmax (int8, sum≈127): ");
    int sum = 0;
    for (int i = 0; i < n; i++) { printf("%d ", out[i]); sum += out[i]; }
    printf("\nSum = %d\n", sum);
    return 0;
}

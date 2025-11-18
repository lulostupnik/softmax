#include "softmax.h"
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * Baseline softmax implementation using floating-point arithmetic.
 * This is a simple, straightforward implementation for comparison.
 *
 * Algorithm:
 * 1. Convert int8 inputs to float
 * 2. Find maximum for numerical stability
 * 3. Compute exp(x[i] - max) for each element
 * 4. Sum all exponentials
 * 5. Normalize and convert back to int8
 *
 * @param x Input array in Q0.7 fixed-point format (-128 to 127)
 * @param n Number of elements
 * @param y Output array in Q0.7 fixed-point format (0 to 127)
 */
void softmax_int8_baseline(const int8_t *x, int n, int8_t *y)
{
    if (n <= 0) return;

    // Step 1: Convert to float and find maximum
    float x_float[n];
    float max_val = -FLT_MAX;

    for (int i = 0; i < n; ++i) {
        x_float[i] = (float)x[i] / 128.0f;  // Convert Q0.7 to float
        if (x_float[i] > max_val) {
            max_val = x_float[i];
        }
    }

    // Step 2: Compute exp(x[i] - max) and sum
    float exp_vals[n];
    float sum = 0.0f;

    for (int i = 0; i < n; ++i) {
        exp_vals[i] = expf(x_float[i] - max_val);
        sum += exp_vals[i];
    }

    // Step 3: Normalize and convert back to int8 (Q0.7)
    for (int i = 0; i < n; ++i) {
        float softmax_val = exp_vals[i] / sum;
        int32_t q7_val = (int32_t)(softmax_val * 128.0f + 0.5f);  // Round

        // Clamp to valid range [0, 127]
        if (q7_val > 127) q7_val = 127;
        if (q7_val < 0) q7_val = 0;

        y[i] = (int8_t)q7_val;
    }
}

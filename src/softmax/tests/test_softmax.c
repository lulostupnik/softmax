/**
 * @file test_softmax.c
 * @brief Test harness for integer-only softmax
 *
 * Tests multiple corner cases and validates against floating-point reference.
 * Computes error metrics: MAE, max error, sum drift, argmax preservation.
 *
 * COMPILE:
 *   gcc -O3 -Wall -Wextra -std=c99 -I../include
 *       test_softmax.c ../src/*.c -o test_softmax -lm
 *
 * RUN:
 *   ./test_softmax
 *
 * DATE: 2025-10-27
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fixedpoint.h"
#include "softmax_int8.h"

/* ========================================================================
 * FLOATING-POINT REFERENCE (TEST CODE ONLY)
 * ======================================================================== */

/**
 * @brief Reference floating-point softmax (stable version)
 *
 * @param x    Input array (int8_t quantized)
 * @param N    Array length
 * @param y    Output probabilities (float)
 */
void softmax_float_reference(const int8_t* x, uint16_t N, float* y) {
    if (N == 0) return;

    /* Find max for numerical stability */
    float max_val = (float)x[0];
    for (uint16_t i = 1; i < N; i++) {
        if ((float)x[i] > max_val) {
            max_val = (float)x[i];
        }
    }

    /* Compute exp(x - max) and sum */
    float sum = 0.0f;
    for (uint16_t i = 0; i < N; i++) {
        y[i] = expf((float)x[i] - max_val);
        sum += y[i];
    }

    /* Normalize */
    if (sum > 0.0f) {
        for (uint16_t i = 0; i < N; i++) {
            y[i] /= sum;
        }
    }
}

/* ========================================================================
 * ERROR METRICS
 * ======================================================================== */

/**
 * @brief Compute error metrics between integer and float softmax
 */
typedef struct
{
    float mae;          /* Mean absolute error */
    float max_err;      /* Maximum absolute error */
    float sum_int;      /* Sum of integer outputs (Q1.7 → float) */
    float sum_float;    /* Sum of float outputs (should be 1.0) */
    uint16_t argmax_int;   /* Argmax of integer output */
    uint16_t argmax_float; /* Argmax of float output */
    int argmax_match;   /* 1 if argmax matches, 0 otherwise */
} error_metrics_t;

/**
 * @brief Compute comprehensive error metrics
 */
error_metrics_t compute_metrics(const uint8_t* y_int, const float* y_float,
                                 uint16_t N) {
    error_metrics_t m = {0};

    if (N == 0) return m;

    float total_err = 0.0f;
    m.max_err = 0.0f;
    m.sum_int = 0.0f;
    m.sum_float = 0.0f;

    uint8_t max_int = y_int[0];
    float max_float = y_float[0];
    m.argmax_int = 0;
    m.argmax_float = 0;

    for (uint16_t i = 0; i < N; i++) {
        /* Convert Q1.7 to float */
        float y_int_f = q1_7_to_float(y_int[i]);

        /* Accumulate errors */
        float err = fabsf(y_int_f - y_float[i]);
        total_err += err;
        if (err > m.max_err) {
            m.max_err = err;
        }

        /* Accumulate sums */
        m.sum_int += y_int_f;
        m.sum_float += y_float[i];

        /* Track argmax */
        if (y_int[i] > max_int) {
            max_int = y_int[i];
            m.argmax_int = i;
        }
        if (y_float[i] > max_float) {
            max_float = y_float[i];
            m.argmax_float = i;
        }
    }

    m.mae = total_err / (float)N;
    m.argmax_match = (m.argmax_int == m.argmax_float) ? 1 : 0;

    return m;
}

/* ========================================================================
 * TEST VECTORS
 * ======================================================================== */

typedef struct
{
    const char* name;
    int8_t data[64];
    uint16_t length;
} test_vector_t;

/* Test case 1: Example from claude.md */
test_vector_t test_single_peak = {
    .name = "Single Peak (claude.md example)",
    .data = {-5, -3, 0, -1, -2, -128, 1, -4},
    .length = 8
};

/* Test case 2: All equal values */
test_vector_t test_uniform = {
    .name = "Uniform (all equal)",
    .data = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    .length = 16
};

/* Test case 3: Heavy negative tail */
test_vector_t test_heavy_tail = {
    .name = "Heavy Negative Tail",
    .data = {10, 5, 0, -10, -20, -30, -40, -50, -60, -70, -80, -90, -100, -110, -120, -128},
    .length = 16
};

/* Test case 4: Alternating peaks */
test_vector_t test_alternating = {
    .name = "Alternating Peaks",
    .data = {20, -20, 20, -20, 20, -20, 20, -20},
    .length = 8
};

/* Test case 5: Edge case - all minimum */
test_vector_t test_all_min = {
    .name = "All Minimum (-128)",
    .data = {-128, -128, -128, -128, -128, -128, -128, -128},
    .length = 8
};

/* Test case 6: Edge case - single max, rest min */
test_vector_t test_single_max = {
    .name = "Single Max, Rest Min",
    .data = {127, -128, -128, -128, -128, -128, -128, -128},
    .length = 8
};

/* Test case 7: Larger sequence (N=64) */
test_vector_t test_large = {
    .name = "Large Sequence (N=64)",
    .data = {
        10, 8, 6, 4, 2, 0, -2, -4, -6, -8, -10, -12, -14, -16, -18, -20,
        -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10,
        5, 3, 1, -1, -3, -5, -7, -9, -11, -13, -15, -17, -19, -21, -23, -25,
        -25, -23, -21, -19, -17, -15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5
    },
    .length = 64
};

/* ========================================================================
 * TEST RUNNER
 * ======================================================================== */

/**
 * @brief Run single test case
 */
void run_test(const test_vector_t* tv) {
    printf("\n========================================\n");
    printf("TEST: %s (N=%d)\n", tv->name, tv->length);
    printf("========================================\n");

    /* Allocate outputs */
    uint8_t* y_int = (uint8_t*)malloc(tv->length * sizeof(uint8_t));
    float* y_float = (float*)malloc(tv->length * sizeof(float));

    if (y_int == NULL || y_float == NULL) {
        printf("ERROR: Memory allocation failed\n");
        free(y_int);
        free(y_float);
        return;
    }

    /* Run integer softmax */
    softmax_int8_row(tv->data, tv->length, y_int, NULL);

    /* Run float reference */
    softmax_float_reference(tv->data, tv->length, y_float);

    /* Compute metrics */
    error_metrics_t metrics = compute_metrics(y_int, y_float, tv->length);

    /* Print inputs */
    printf("\nInputs (first 16): ");
    for (uint16_t i = 0; i < tv->length && i < 16; i++) {
        printf("%4d ", tv->data[i]);
    }
    if (tv->length > 16) printf("...");
    printf("\n");

    /* Print outputs (first 16) */
    printf("\nInteger outputs (Q1.7, first 16): ");
    for (uint16_t i = 0; i < tv->length && i < 16; i++) {
        printf("%3d ", y_int[i]);
    }
    if (tv->length > 16) printf("...");
    printf("\n");

    printf("Float outputs (first 16):          ");
    for (uint16_t i = 0; i < tv->length && i < 16; i++) {
        printf("%.3f ", y_float[i]);
    }
    if (tv->length > 16) printf("...");
    printf("\n");

    /* Print metrics */
    printf("\n--- Metrics ---\n");
    printf("Sum (integer):  %.6f  (expected ~1.0)\n", metrics.sum_int);
    printf("Sum (float):    %.6f  (expected 1.0)\n", metrics.sum_float);
    printf("Sum error:      %.6f  (%%)\n",
           fabsf(metrics.sum_int - 1.0f) * 100.0f);
    printf("MAE:            %.6f\n", metrics.mae);
    printf("Max error:      %.6f\n", metrics.max_err);
    printf("Argmax (int):   %d\n", metrics.argmax_int);
    printf("Argmax (float): %d\n", metrics.argmax_float);
    printf("Argmax match:   %s\n", metrics.argmax_match ? "YES ✓" : "NO ✗");

    /* Validation checks */
    int pass = 1;

    if (fabsf(metrics.sum_int - 1.0f) > 0.02f) {  /* Within 2% */
        printf("\n⚠ WARNING: Sum drift > 2%%\n");
        pass = 0;
    }

    if (!metrics.argmax_match) {
        printf("\n✗ FAIL: Argmax mismatch!\n");
        pass = 0;
    }

    if (metrics.mae > 0.05f) {  /* MAE > 5% */
        printf("\n⚠ WARNING: High MAE (%.2f%% avg error)\n", metrics.mae * 100.0f);
    }

    if (pass) {
        printf("\n✓ PASS: All checks passed\n");
    }

    /* Cleanup */
    free(y_int);
    free(y_float);
}

/* ========================================================================
 * MAIN
 * ======================================================================== */

int main(void) {
    printf("╔════════════════════════════════════════════════════════╗\n");
    printf("║  Integer-Only Softmax Test Suite                      ║\n");
    printf("║  Hardware-friendly base-2 exp with LUT                ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n");

    /* Run all test cases */
    run_test(&test_single_peak);
    run_test(&test_uniform);
    run_test(&test_heavy_tail);
    run_test(&test_alternating);
    run_test(&test_all_min);
    run_test(&test_single_max);
    run_test(&test_large);

    printf("\n========================================\n");
    printf("All tests completed.\n");
    printf("========================================\n");

    return 0;
}

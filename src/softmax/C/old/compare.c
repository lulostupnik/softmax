#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

// ===================== Constants for Custom Algo =====================
#define K_A 5
#define SHIFT_BUCK 16 // (1 << (K_A - 1))
#define ONE_Q7 128
#define LOG2E 1.44269504088896340736 // log2(e)


// ===================== Float Reference & Utils =====================

static inline double ita_epsilon_B8(void) {
    return (double)8.0 / (256.0 * LOG2E); // ≈ 0.02162
}

static void quantize_row_ita_eps(const float *x, int n, int8_t *xq) {
    const double eps = ita_epsilon_B8();
    for (int i = 0; i < n; ++i) {
        double q = nearbyint(x[i] / eps);
        if (q < -128.0) q = -128.0;
        if (q > 127.0)  q = 127.0;
        xq[i] = (int8_t)q;
    }
}

// Standard Float Softmax (The Ground Truth)
static void softmax_ref_q07_from_float(const float *x, int n, int8_t *y_ref) {
    if (n <= 0) return;
    float m = x[0];
    for (int i = 1; i < n; ++i) if (x[i] > m) m = x[i];

    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += exp((double)x[i] - m);

    for (int i = 0; i < n; ++i) {
        double p = exp((double)x[i] - m) / sum;
        // Scale 0..1 to 0..127
        int q = (int)lrint(p * 127.0);
        if (q < 0) q = 0;
        if (q > 127) q = 127;
        y_ref[i] = (int8_t)q;
    }
}

// ===================== Test Utilities =====================

static int sum_q7(const int8_t *y, int n) {
    int s = 0;
    for (int i = 0; i < n; ++i) s += (int)y[i];
    return s;
}

static int max_abs_diff(const int8_t *a, const int8_t *b, int n) {
    int m = 0;
    for (int i = 0; i < n; ++i) {
        int d = (int)a[i] - (int)b[i];
        if (d < 0) d = -d;
        if (d > m) m = d;
    }
    return m;
}

static void print_vec_i8(const char *name, const int8_t *v, int n) {
    printf("%-12s: [", name);
    for (int i = 0; i < n; ++i) {
        printf("%3d", (int)v[i]);
        if (i + 1 < n) printf(", ");
    }
    printf("]\n");
}

// ===================== Comparison Runner =====================

static void run_case_with_ita_quant(const char *title, const float *xf, int n)
{
    int8_t xq[512];
    int8_t y_custom[512];
    int8_t y_ref[512];

    if (n > 512) n = 512;

    // 1. Quantize Float -> Int8 (Simulating hardware input)
    quantize_row_ita_eps(xf, n, xq);

    // 2. Run CUSTOM Algorithm
    softmax_int8_optionA(xq, n, y_custom);

    // 3. Run FLOAT Reference (Ground Truth)
    softmax_ref_q07_from_float(xf, n, y_ref);

    // 4. Print Results
    printf("\n==========================================================\n");
    printf("TEST: %s (n=%d)\n", title, n);
    printf("==========================================================\n");

    printf("Input Float : [");
    for (int i = 0; i < n; ++i) { printf("%.2f%s", xf[i], (i+1<n)?", ":""); }
    printf("]\n");

    print_vec_i8("Input Int8", xq, n);
    printf("----------------------------------------------------------\n");
    print_vec_i8("Custom Algo", y_custom, n);
    print_vec_i8("Float Ref", y_ref, n);
    printf("----------------------------------------------------------\n");

    int sum_custom = sum_q7(y_custom, n);
    int sum_ref = sum_q7(y_ref, n);
    int max_diff = max_abs_diff(y_custom, y_ref, n);

    printf("Sum Check   : Custom=%d, Ref=%d (Target ~127)\n", sum_custom, sum_ref);
    printf("Max Abs Diff: %d\n", max_diff);

    if (max_diff <= 2) {
        printf("RESULT: [PASS] (Approximation is within acceptable quantization error)\n");
    } else {
        printf("RESULT: [FAIL] (Approximation diverged too much)\n");
    }
}

// ===================== Main =====================
int main(void)
{
    // Case 1: Standard distribution
    float xf1[8] = {-4.0f, -2.0f, -1.0f, -0.2f, 0.0f, 0.3f, 0.8f, 1.5f};
    run_case_with_ita_quant("Standard Distribution", xf1, 8);

    // Case 2: Peaky distribution (One dominant value)
    // This tests if the max search works correctly
    float xf2[8] = {-10.0f, -10.0f, -10.0f, -10.0f, -10.0f, -5.0f, 0.0f, 8.0f};
    run_case_with_ita_quant("Peaky (Dominant Max)", xf2, 8);

    // Case 3: Flat distribution (All values similar)
    // This tests precision of the sum accumulation
    float xf3[8] = {0.1f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f};
    run_case_with_ita_quant("Flat Distribution", xf3, 8);

    return 0;
}

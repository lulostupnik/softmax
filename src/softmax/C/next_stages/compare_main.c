#include <math.h>
#include <time.h>
#include "softmax.h"



static inline double ita_epsilon_B8(void) {
    // ITA ε (B=8): ε = B/(2^B·log2 e); ε′ = 1/32 ⇒ bucket=32
    return (double)8.0 / (256.0 * LOG2E); // ≈ 0.02162
}

// Quantize a float row to int8 using the ITA epsilon.
// Then (optionally) clip extreme tails (“non-survivors”) that would underflow.
static void quantize_row_ita_eps(const float *x, int n, int8_t *xq) {
    const double eps = ita_epsilon_B8();
    // quantize
    for (int i = 0; i < n; ++i) {
        double q = nearbyint(x[i] / eps);
        if (q < -128.0) q = -128.0;
        if (q > 127.0)  q = 127.0;
        xq[i] = (int8_t)q;
    }
#ifdef ITA_CLIP_SURVIVORS
    // Optional survivor clipping: drop entries whose top-3 MSBs would kill them
    int8_t m = xq[0];
    for (int i = 1; i < n; ++i) if (xq[i] > m) m = xq[i];
    for (int i = 0; i < n; ++i) {
        uint8_t d = (uint8_t)(m - xq[i]);
        uint8_t bucket3 = (uint8_t)(d >> 5); // MSB bucket (3 bits)
        if (bucket3 >= 8) xq[i] = (int8_t)(m - (int8_t)255); // force very low
    }
#endif
}

// ===================== Float softmax reference (to Q0.7) =====================  //
static void softmax_ref_q07_from_float(const float *x, int n, int8_t *y_ref) {
    if (n <= 0) return;
    float m = x[0];
    for (int i = 1; i < n; ++i) if (x[i] > m) m = x[i];
    float sum = 0.f;
    for (int i = 0; i < n; ++i) sum += expf(x[i] - m);
    for (int i = 0; i < n; ++i) {
        float p = expf(x[i] - m) / sum;      // [0,1]
        int q = (int)lrintf(p * 127.f);      // Q0.7
        if (q < 0) q = 0; if (q > 127) q = 127;
        y_ref[i] = (int8_t)q;
    }
}

//



// ===================== Test utilities =====================
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
    printf("%s: [", name);
    for (int i = 0; i < n; ++i) {
        printf("%d", (int)v[i]);
        if (i + 1 < n) printf(", ");
    }
    printf("]\n");
}

// Run a float → ITA-quant → int-softmax (A & B) case
static void run_case_with_ita_quant(const char *title,
                                    const float *xf, int n)
{
    int8_t xq[512], yA[512], yB[512], yR[512];
    if (n > 512) n = 512;

    // 1) ITA quantize
    quantize_row_ita_eps(xf, n, xq);

    // 2) Integer softmax on xq
    softmax_int8_optionA(xq, n, yA);
    softmax_int8_optionB(xq, n, yB);

    // 3) Float reference directly on xf
    softmax_ref_q07_from_float(xf, n, yR);

    // 4) Print/compare
    printf("\n=== %s (n=%d) ===\n", title, n);

    printf("x_float : [");
    for (int i = 0; i < n; ++i) { printf("%.3f%s", xf[i], (i+1<n)?", ":""); }
    printf("]\n");

    print_vec_i8("x_q(ITA)", xq, n);
    print_vec_i8("optA_int", yA, n);
    print_vec_i8("optB_int", yB, n);
    print_vec_i8("ref_soft", yR, n);

    printf("sum A=%d, B=%d, ref=%d | max_abs_diff A=%d, B=%d\n",
           sum_q7(yA,n), sum_q7(yB,n), sum_q7(yR,n),
           max_abs_diff(yA,yR,n), max_abs_diff(yB,yR,n));
}

// ===================== Main =====================
int main(void)
{
    srand((unsigned)time(NULL));
    // A few realistic rows (logits after some linear layers)
    float xf1[8] = {-4.0f, -2.0f, -1.0f, -0.2f, 0.0f, 0.3f, 0.8f, 1.5f};
    float xf2[8] = {-6.0f,-5.0f,-3.0f,-2.5f,-2.0f,-1.0f,0.0f, 6.0f};
    float xf3[8] = { 0.0f, -0.5f, 1.2f, -2.1f, 0.7f, 2.3f, -1.4f, 0.2f};

    run_case_with_ita_quant("ITA-quant test 1", xf1, 8);
    run_case_with_ita_quant("ITA-quant test 2 (peaky)", xf2, 8);
    run_case_with_ita_quant("ITA-quant test 3", xf3, 8);

    // Random demo: generate float logits ~N(0,1), then ITA-quant
    for (int t = 0; t < 3; ++t) {
        float xr[16];
        for (int i = 0; i < 16; ++i) {
            // simple approx to N(0,1): sum of 12 uniforms - 6
            double s = 0.0;
            for (int k = 0; k < 12; ++k) s += (double)rand() / (double)RAND_MAX;
            xr[i] = (float)(s - 6.0);
        }
        char title[64];
        snprintf(title, sizeof(title), "ITA-quant random %d", t+1);
        run_case_with_ita_quant(title, xr, 16);
    }
    return 0;
}
